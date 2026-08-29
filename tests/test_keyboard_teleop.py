from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "ros2_ws"
    / "src"
    / "livifuser_command_watchdog"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_command_watchdog.keyboard_policy import (  # noqa: E402
    DEFAULT_RELEASE_AWARE_KEYS,
    ZERO_COMMAND,
    KeyboardCommand,
    ReleaseAwareKeyboard,
    command_for_key,
    validate_allowed_motion_keys,
)


class KeyboardPolicyTests(unittest.TestCase):
    def test_planar_bindings_have_expected_signs(self) -> None:
        expected = {
            "i": KeyboardCommand(0.08, 0.0),
            "u": KeyboardCommand(0.08, 0.40),
            "o": KeyboardCommand(0.08, -0.40),
            "j": KeyboardCommand(0.0, 0.40),
            "l": KeyboardCommand(0.0, -0.40),
            ",": KeyboardCommand(-0.08, 0.0),
            ".": KeyboardCommand(-0.08, 0.40),
            "m": KeyboardCommand(-0.08, -0.40),
        }
        for key, command in expected.items():
            with self.subTest(key=key):
                self.assertEqual(command_for_key(key), command)

    def test_k_and_every_unknown_key_stop(self) -> None:
        for key in ("k", "q", "z", "I", " ", ""):
            with self.subTest(key=key):
                self.assertEqual(command_for_key(key), ZERO_COMMAND)

    def test_custom_limits_scale_every_binding(self) -> None:
        self.assertEqual(
            command_for_key("o", linear_mps=0.05, angular_radps=0.25),
            KeyboardCommand(0.05, -0.25),
        )

    def test_invalid_limits_are_rejected(self) -> None:
        for linear, angular in (
            (0.0, 0.4),
            (0.11, 0.4),
            (0.08, 0.0),
            (0.08, 0.51),
            (math.nan, 0.4),
            (0.08, math.inf),
        ):
            with self.subTest(linear=linear, angular=angular), self.assertRaises(
                ValueError
            ):
                command_for_key("i", linear_mps=linear, angular_radps=angular)


class ReleaseAwareKeyboardTests(unittest.TestCase):
    def test_default_final_protocol_has_no_reverse_or_in_place_turn(self) -> None:
        self.assertEqual(DEFAULT_RELEASE_AWARE_KEYS, frozenset({"i", "u", "o"}))

    def test_key_release_immediately_returns_zero(self) -> None:
        keyboard = ReleaseAwareKeyboard()
        self.assertEqual(keyboard.key_down("u"), KeyboardCommand(0.08, 0.40))
        self.assertEqual(keyboard.key_up("u"), ZERO_COMMAND)
        self.assertEqual(keyboard.command, ZERO_COMMAND)

    def test_key_repeat_is_idempotent(self) -> None:
        keyboard = ReleaseAwareKeyboard()
        keyboard.key_down("i")
        keyboard.key_down("i")
        self.assertEqual(keyboard.pressed, frozenset({"i"}))
        self.assertEqual(keyboard.command, KeyboardCommand(0.08, 0.0))

    def test_second_motion_key_fails_to_zero_without_resuming(self) -> None:
        keyboard = ReleaseAwareKeyboard()
        keyboard.key_down("u")
        self.assertEqual(keyboard.key_down("i"), ZERO_COMMAND)
        self.assertEqual(keyboard.key_up("i"), ZERO_COMMAND)
        self.assertEqual(keyboard.pressed, frozenset())

    def test_unknown_stop_and_focus_clear_fail_to_zero(self) -> None:
        keyboard = ReleaseAwareKeyboard()
        for stop in ("k", "escape", "x", ""):
            with self.subTest(stop=stop):
                keyboard.key_down("o")
                self.assertEqual(keyboard.key_down(stop), ZERO_COMMAND)
                self.assertEqual(keyboard.pressed, frozenset())
        keyboard.key_down("i")
        self.assertEqual(keyboard.clear(), ZERO_COMMAND)

    def test_allowed_key_validation_refuses_reverse_and_empty_sets(self) -> None:
        self.assertEqual(validate_allowed_motion_keys(["I", "u"]), frozenset({"i", "u"}))
        for keys in ([], [","], ["m"], ["i", "o", "reverse"]):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                validate_allowed_motion_keys(keys)


if __name__ == "__main__":
    unittest.main()
