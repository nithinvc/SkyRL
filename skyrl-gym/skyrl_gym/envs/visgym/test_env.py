"""Smoke test for the VisGym → BaseTextEnv wrapper.

Requires both `gymnasium` (VisGym) and `skyrl_gym` to be installed:
    pip install -e VisGym/
    pip install -e skyrl-gym/

Run: python -m skyrl_gym.envs.visgym.test_env
"""

import base64

from skyrl_gym.envs.visgym.env import VisGymEnv
from skyrl_gym.envs.visgym.utils import extract_action


def test_action_extraction():
    """Test that extract_action correctly parses various VLM outputs."""
    # Bare action
    action, matched = extract_action("('move', 0)")
    assert action == "('move', 0)" and matched
    # Reasoning + action
    action, matched = extract_action("I should go right. ('move', 0)")
    assert action == "('move', 0)" and matched
    # Action in the middle of text
    action, matched = extract_action("Let me think... ('stop', 'stop') done!")
    assert action == "('stop', 'stop')" and matched
    # Nested tuple (jigsaw swap)
    action, matched = extract_action("Swapping pieces. ('swap', ((0, 0), (1, 1)))")
    assert action.startswith("('swap',") and matched
    # No action found — returns stripped input, matched=False
    action, matched = extract_action("I have no idea what to do")
    assert action == "I have no idea what to do" and not matched
    # Whitespace variations
    action, matched = extract_action("( 'move' , 2 )")
    assert action == "( 'move' , 2 )" and matched
    print("[PASS] Action extraction tests")


def test_solver_trajectory():
    """Run the built-in solver through the wrapper and verify reward=1.0."""
    extras = {
        "visgym_env_id": "maze_2d/easy",
        "seed": 42,
        "max_turns": 50,  # generous limit for solver
    }
    env = VisGymEnv(extras=extras)

    # Initialize
    messages, metadata = env.init([])

    # Verify initial message structure
    assert len(messages) == 1, f"Expected 1 initial message, got {len(messages)}"
    msg = messages[0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list), "Content should be a list (multimodal)"
    assert len(msg["content"]) == 2, "Content should have text + image parts"
    assert msg["content"][0]["type"] == "text"
    assert msg["content"][1]["type"] == "image_url"

    # Verify image is valid base64
    url = msg["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    b64_data = url.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_data)
    assert len(img_bytes) > 0, "Image should not be empty"
    print(f"[INFO] Initial prompt length: {len(msg['content'][0]['text'])} chars")
    print(f"[INFO] Initial image size: {len(img_bytes)} bytes")

    # Get solver trajectory from the underlying VisGym env
    solver_actions = env.visgym_env.solve()
    print(f"[INFO] Solver trajectory: {len(solver_actions)} actions")

    # Execute solver trajectory
    final_reward = 0.0
    for i, action_str in enumerate(solver_actions):
        result = env.step(action_str)
        final_reward = result["reward"]
        done = result["done"]

        if not done:
            obs = result["observations"]
            assert len(obs) == 1, f"Step {i}: expected 1 observation, got {len(obs)}"
            assert obs[0]["role"] == "user"
            assert isinstance(obs[0]["content"], list)
        else:
            assert result["observations"] == [], "Final step should have empty observations"
            break

    assert final_reward == 1.0, f"Solver should get reward 1.0, got {final_reward}"
    assert done, "Solver trajectory should end with done=True"
    print(f"[PASS] Solver trajectory: {env.step_count} steps, reward={final_reward}")

    env.close()


def test_reasoning_with_action():
    """Test that VLM-style reasoning + action works through the wrapper."""
    extras = {
        "visgym_env_id": "maze_2d/easy",
        "seed": 42,
        "max_turns": 5,
    }
    env = VisGymEnv(extras=extras)
    env.init([])

    # Simulate VLM output with reasoning
    vlm_output = "Looking at the maze, I can see the target is to the right. I'll move right. ('move', 0)"
    result = env.step(vlm_output)

    assert result["metadata"]["extracted_action"] == "('move', 0)"
    assert env.parse_failures == 0, "Should have successfully extracted action"
    print(f"[PASS] Reasoning + action parsing")

    env.close()


def test_parse_failure():
    """Test that unparseable VLM output is handled gracefully."""
    extras = {
        "visgym_env_id": "maze_2d/easy",
        "seed": 42,
        "max_turns": 5,
    }
    env = VisGymEnv(extras=extras)
    env.init([])

    # Garbage output with no tuple pattern
    result = env.step("I don't know what to do here")

    assert env.parse_failures == 1
    assert not result["done"], "Parse failure should not end episode"
    # VisGym should provide env_feedback about the parse error
    feedback = result["metadata"].get("env_feedback", "")
    assert feedback, f"Expected env_feedback on parse failure, got empty string"
    print(f"[PASS] Parse failure handled gracefully. Feedback: {feedback[:80]}...")

    env.close()


def test_max_turns_truncation():
    """Test that episode ends when max_turns is reached."""
    extras = {
        "visgym_env_id": "maze_2d/easy",
        "seed": 42,
        "max_turns": 2,
    }
    env = VisGymEnv(extras=extras)
    env.init([])

    result1 = env.step("('move', 0)")
    assert not result1["done"], "Should not be done after 1 step with max_turns=2"

    result2 = env.step("('move', 0)")
    assert result2["done"], "Should be done after 2 steps with max_turns=2"
    print(f"[PASS] Max turns truncation")

    env.close()


def test_metrics():
    """Test that metrics are returned correctly."""
    extras = {
        "visgym_env_id": "maze_2d/easy",
        "seed": 42,
        "max_turns": 5,
    }
    env = VisGymEnv(extras=extras)
    env.init([])
    env.step("('move', 0)")
    env.step("garbage text")

    metrics = env.get_metrics()
    assert metrics["step_count"] == 2
    assert metrics["parse_failures"] == 1
    assert metrics["visgym_env_id"] == "maze_2d/easy"
    print(f"[PASS] Metrics: {metrics}")

    env.close()


if __name__ == "__main__":
    test_action_extraction()
    test_solver_trajectory()
    test_reasoning_with_action()
    test_parse_failure()
    test_max_turns_truncation()
    test_metrics()
    print("\n[ALL TESTS PASSED]")
