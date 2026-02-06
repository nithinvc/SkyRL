from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    orjson = None
from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv

from slime.rollout.rm_hub import grade_answer_verl
from slime.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Matches the JSON payload emitted between <tool_call> ... </tool_call> tags.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Accept either name; verl uses `calc_geo3k_reward` while the instruction refers to `calc_score`.
SUPPORTED_TOOL_NAMES = {"calc_score", "calc_geo3k_reward"}
# from https://github.com/agentica-project/deepscaler/blob/e6080ccd974eb64bd3430f0b36108244a6fee330/deepscaler/rewards/math_utils/utils.py
"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""
import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser


# Dan Hendrycks' code
def mathd_normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == f"{a}/{b}"
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except Exception:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except Exception:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems, strict=False):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def grade_answer_verl(solution_str, ground_truth):
    if not ground_truth:
        return False
    ground_truth = str(ground_truth)
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth)
    given_answer = extract_answer(solution_str)
    if given_answer is None:
        return False
    return grade_answer_mathd(given_answer, ground_truth) or grade_answer_sympy(given_answer, ground_truth)

class Geo3kEnv(BaseInteractionEnv):
    """
    Minimal interaction environment for multi-turn geo3k with a scoring tool.

    The model is expected to emit a <tool_call>{...}</tool_call> payload that includes
    an `answer` argument. We run the math reward checker against the ground truth and
    return the score as the next observation. The episode ends immediately after each
    step; responses are provided but no further turns are taken.
    """

    def __init__(self, *, ground_truth: str | None = None, max_turns: int | None = None):
        self.ground_truth = str(ground_truth) if ground_truth is not None else None
        self.tool_calls: list[dict[str, Any]] = []
        self.last_tool_score: float | None = None
        self.turn = 0
        self.max_turns = max_turns

    def reset(self):
        self.tool_calls.clear()
        self.last_tool_score = None
        self.turn = 0
        # No initial observation is needed; the question lives in the prompt.
        observation: dict[str, Any] = {}
        reset_info = {"ground_truth_available": self.ground_truth is not None}
        return observation, reset_info

    def close(self):
        """No resources to release."""
        return

    def _extract_tool_call(self, text: str) -> dict[str, Any] | None:
        """
        Parse the latest tool call payload from the assistant response.
        Supports the <tool_call>{...}</tool_call> convention used in the
        SGLang multi-turn templates. Tool tags are mandatory.
        """
        matches = list(TOOL_CALL_RE.finditer(text))
        raw_json = None
        if matches:
            raw_json = matches[-1].group(1).strip()

        if raw_json is None:
            return None

        payload = self._parse_tool_payload(raw_json)
        if payload is None:
            return None

        name = payload.get("name") or payload.get("function", {}).get("name")
        arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Tool call arguments are not valid JSON; rejecting tool call.")
                return None

        if not name:
            return None
        return {"name": name, "arguments": arguments}

    def _score_answer(self, answer: str) -> float:
        """
        Use the same logic as the single-turn math reward model.
        We accept either boxed or raw numeric strings by retrying with a boxed wrapper.
        """
        if not self.ground_truth:
            return 0.0

        answer = answer.strip()
        candidates = [answer]
        if "\\boxed" not in answer:
            candidates.append(f"\\boxed{{{answer}}}")

        for candidate in candidates:
            try:
                if grade_answer_verl(candidate, self.ground_truth):
                    return 1.0
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("grade_answer_verl failed on %s: %s", candidate, exc)
                continue
        return 0.0

    def _extract_answer_from_text(self, text: str) -> str | None:
        """
        Prefer a concise answer by pulling the last \\boxed{} chunk; fall back to the last
        non-empty line (capped) to avoid echoing the whole response body.
        """
        boxed = extract_boxed_answer(text)
        if boxed:
            return str(boxed).strip()
        for line in reversed(text.splitlines()):
            cleaned = line.strip()
            if cleaned:
                return cleaned[:512]
        trimmed = text.strip()
        return trimmed[:512] if trimmed else None

    def _extract_balanced_json(self, text: str, start: int) -> str | None:
        """
        Best-effort balanced brace extraction starting at `start` (index of an opening '{').
        Keeps string-awareness to avoid terminating inside quoted braces.
        """
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "\\" and not escaped:
                escaped = True
                continue
            if ch == '"' and not escaped:
                in_string = not in_string
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
            escaped = False
        return None

    def _build_tool_feedback(self, score: float, parsed_answer: str) -> str:
        """
        Provide concise feedback for the model to continue reasoning.
        """
        turn_idx = self.turn - 1  # zero-based
        # Send the final reminder one turn before the true last turn so the model sees it in time.
        last_warning_turn = None
        if self.max_turns is not None:
            if self.max_turns >= 2:
                last_warning_turn = self.max_turns - 2
            else:
                last_warning_turn = self.max_turns - 1
        is_final_turn = last_warning_turn is not None and turn_idx >= last_warning_turn

        if score == 1.0:
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' matches the reference. "
                "You can now stop reasoning and provide the final solution in \\boxed{}."
            )
        if score == 0.0:
            if is_final_turn:
                return (
                    f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
                    "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary. "
                    "Since you only have one chance to answer, don't call tool again. You should provide your final answer in the form below Answer: \\boxed{$Answer} where $Answer is your fiinal answer to this problem."
                )
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
                "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary."
            )

    # Called during rollout after receiving a model response
    def step(self, response_text: str):
        self.turn += 1
        is_final_turn = self.max_turns is not None and self.turn >= self.max_turns
        tool_call = self._extract_tool_call(response_text)
        info: dict[str, Any] = {"tool_call": deepcopy(tool_call)}

        if not tool_call:
            info["tool_executed"] = False
            obs = {
                "obs_str": "No tool call detected; ending the episode.",
                "role": "tool",
            }
            return obs, True, info

        name = (tool_call.get("name") or "").strip()
        arguments = tool_call.get("arguments") or {}
        if name not in SUPPORTED_TOOL_NAMES:
            obs = {
                "obs_str": (
                    f"Tool `{name}` is not supported. "
                    'Call `calc_score` (or `calc_geo3k_reward`) via <tool_call>{"name": "calc_score", "arguments": {"answer": "<digits>"}}</tool_call> (format must be <tool_call>(JSON)</tool_call>)'
                    "to check your solution."
                ),
                "role": "tool",
            }
            info["tool_executed"] = False
            return obs, is_final_turn, info

        raw_answer = arguments.get("answer", None)
        parsed_answer = "" if raw_answer is None else str(raw_answer)
        if not parsed_answer.strip():
            obs = {
                "obs_str": (
                    "Tool call detected but no `answer` was provided. "
                    'Call `calc_score` (or `calc_geo3k_reward`) via <tool_call>{"name": "calc_score", "arguments": {"answer": "<digits>"}}</tool_call> '
                    "to check your solution."
                ),
                "role": "tool",
            }
            info["tool_executed"] = False
            info["answer_missing"] = True
            return obs, is_final_turn, info

        score = self._score_answer(parsed_answer)
        self.last_tool_score = score
        tool_record = {"name": name, "answer": parsed_answer, "score": score}
        self.tool_calls.append(tool_record)
        info.update(tool_record)
        info["tool_executed"] = True

        obs = {
            "obs_str": self._build_tool_feedback(score, parsed_answer),
            "role": "tool",
            "tool_score": score,
        }

        return obs, is_final_turn, info

    def _parse_tool_payload(self, raw_json: str) -> dict[str, Any] | None:
        """Parse tool payload strictly as JSON. Malformed payloads are rejected."""
        loader = orjson.loads if orjson is not None else json.loads
        try:
            return loader(raw_json)
        except Exception as exc:
            logger.warning("Failed to decode tool call payload: %s", exc)
            return None


def _extract_ground_truth(sample: Sample | None) -> str | None:
    """Resolve the ground-truth answer from label or metadata."""
    if sample is None:
        return None
    if sample.label is not None:
        return str(sample.label)
    # metadata = sample.metadata
    # for key in ("answer", "ground_truth", "label"):
    #     if key in metadata and metadata[key] is not None:
    #         return str(metadata[key])
    return None


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> Geo3kEnv:
    """
    Construct a Geo3kEnv. Ground truth is pulled from sample.label or metadata.
    """
    ground_truth = _extract_ground_truth(sample)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    if ground_truth is None:
        logger.warning("Ground truth answer missing; calc_score tool will always return 0.")
    return Geo3kEnv(ground_truth=ground_truth, max_turns=max_turns)