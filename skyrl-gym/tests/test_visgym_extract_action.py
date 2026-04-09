import pytest

from skyrl_gym.envs.visgym.utils import extract_action


# ---------------------------------------------------------------------------
# Simple / flat payloads (no nesting)
# ---------------------------------------------------------------------------


class TestFlatPayloads:
    def test_integer_payload(self):
        result, matched = extract_action("('move', 0)")
        assert matched
        assert result == "('move', 0)"

    def test_string_payload(self):
        result, matched = extract_action("('stop', 'stop')")
        assert matched
        assert result == "('stop', 'stop')"

    def test_double_quoted_action(self):
        result, matched = extract_action('("move", 3)')
        assert matched
        assert eval(result) == ("move", 3)

    def test_negative_integer(self):
        result, matched = extract_action("('move', -1)")
        assert matched
        assert eval(result) == ("move", -1)

    def test_float_payload(self):
        result, matched = extract_action("('guess', 3.14)")
        assert matched
        assert eval(result) == ("guess", 3.14)


# ---------------------------------------------------------------------------
# Nested tuple payloads (the main bug the old regex had)
# ---------------------------------------------------------------------------


class TestNestedTuples:
    def test_pair_payload(self):
        result, matched = extract_action("('mark', (0.5, 0.5))")
        assert matched
        assert eval(result) == ("mark", (0.5, 0.5))

    def test_triple_payload(self):
        result, matched = extract_action("('place', (2, 1, 3))")
        assert matched
        assert eval(result) == ("place", (2, 1, 3))

    def test_deeply_nested(self):
        result, matched = extract_action("('swap', ((0,0),(1,1)))")
        assert matched
        assert eval(result) == ("swap", ((0, 0), (1, 1)))

    def test_move_with_direction_tuple(self):
        result, matched = extract_action("('move', (1, 2))")
        assert matched
        assert eval(result) == ("move", (1, 2))


# ---------------------------------------------------------------------------
# List payloads
# ---------------------------------------------------------------------------


class TestListPayloads:
    def test_list_of_ints(self):
        result, matched = extract_action("('reorder', [0, 1, 2, 3])")
        assert matched
        assert eval(result) == ("reorder", [0, 1, 2, 3])

    def test_list_of_tuples(self):
        result, matched = extract_action("('multi', [(0,1), (2,3)])")
        assert matched
        assert eval(result) == ("multi", [(0, 1), (2, 3)])


# ---------------------------------------------------------------------------
# Surrounding reasoning text
# ---------------------------------------------------------------------------


class TestSurroundingText:
    def test_reasoning_before(self):
        text = "I think I should move right.\n('move', 1)"
        result, matched = extract_action(text)
        assert matched
        assert eval(result) == ("move", 1)

    def test_reasoning_before_and_after(self):
        text = "Let me mark the center.\n('mark', (0.5, 0.5))\nDone."
        result, matched = extract_action(text)
        assert matched
        assert eval(result) == ("mark", (0.5, 0.5))

    def test_long_reasoning_block(self):
        text = (
            "Looking at the maze, I can see the path goes right then down. "
            "The agent is at position (2, 3) and the target is at (5, 7). "
            "I need to move down first.\n\n('move', 2)\n\n"
            "This should get me closer to the target."
        )
        result, matched = extract_action(text)
        assert matched
        assert eval(result) == ("move", 2)


# ---------------------------------------------------------------------------
# Markdown / backtick wrapping (fallback #2)
# ---------------------------------------------------------------------------


class TestMarkdownStripping:
    def test_inline_backticks(self):
        text = "I'll do `('mark', (0.5, 0.5))`"
        result, matched = extract_action(text)
        assert matched
        assert eval(result) == ("mark", (0.5, 0.5))

    def test_code_fence(self):
        text = "Here is my action:\n```python\n('move', 0)\n```"
        result, matched = extract_action(text)
        assert matched
        assert eval(result) == ("move", 0)

    def test_code_fence_nested_tuple(self):
        text = "```\n('place', (2, 1, 3))\n```"
        result, matched = extract_action(text)
        assert matched
        assert eval(result) == ("place", (2, 1, 3))


# ---------------------------------------------------------------------------
# Unquoted action names (fallback #3)
# ---------------------------------------------------------------------------


class TestUnquotedActionNames:
    def test_unquoted_simple(self):
        result, matched = extract_action("(move, 0)")
        assert matched
        assert eval(result) == ("move", 0)

    def test_unquoted_nested(self):
        result, matched = extract_action("(mark, (0.5, 0.3))")
        assert matched
        assert eval(result) == ("mark", (0.5, 0.3))

    def test_unquoted_stop(self):
        result, matched = extract_action("(stop, 'stop')")
        assert matched
        parsed = eval(result)
        assert parsed[0] == "stop"


# ---------------------------------------------------------------------------
# JSON fallback (fallback #4)
# ---------------------------------------------------------------------------


class TestJsonFallback:
    def test_json_simple(self):
        text = '{"action": "move", "args": 0}'
        result, matched = extract_action(text)
        assert matched
        assert eval(result) == ("move", 0)

    def test_json_with_list_args(self):
        text = '{"action": "reorder", "args": [0, 1, 2, 3]}'
        result, matched = extract_action(text)
        assert matched
        parsed = eval(result)
        assert parsed[0] == "reorder"
        assert list(parsed[1]) == [0, 1, 2, 3]

    def test_json_with_tuple_like_args(self):
        text = '{"action": "mark", "args": [0.5, 0.5]}'
        result, matched = extract_action(text)
        assert matched
        parsed = eval(result)
        assert parsed[0] == "mark"

    def test_json_in_reasoning(self):
        text = 'I will mark the center. {"action": "mark", "args": [0.5, 0.5]} Done.'
        result, matched = extract_action(text)
        assert matched
        assert eval(result)[0] == "mark"


# ---------------------------------------------------------------------------
# Malformed / unrecoverable input
# ---------------------------------------------------------------------------


class TestMalformedInput:
    def test_empty_string(self):
        result, matched = extract_action("")
        assert not matched

    def test_pure_text(self):
        result, matched = extract_action("I don't know what to do.")
        assert not matched

    def test_incomplete_tuple(self):
        result, matched = extract_action("('move',")
        assert not matched

    def test_no_payload(self):
        result, matched = extract_action("('move')")
        assert not matched

    def test_returns_stripped_raw_on_failure(self):
        raw = "  some random text  "
        result, matched = extract_action(raw)
        assert not matched
        assert result == "some random text"


# ---------------------------------------------------------------------------
# Whitespace / formatting variations
# ---------------------------------------------------------------------------


class TestWhitespaceVariations:
    def test_extra_spaces(self):
        result, matched = extract_action("(  'move'  ,  0  )")
        assert matched
        assert eval(result) == ("move", 0)

    def test_newline_in_tuple(self):
        result, matched = extract_action("('mark',\n(0.5, 0.5))")
        assert matched
        assert eval(result) == ("mark", (0.5, 0.5))

    def test_tabs(self):
        result, matched = extract_action("(\t'stop'\t,\t'stop'\t)")
        assert matched
        assert eval(result) == ("stop", "stop")
