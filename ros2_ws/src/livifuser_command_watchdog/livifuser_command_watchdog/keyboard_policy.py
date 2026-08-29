"""Pure key-to-intent policy for the acquisition keyboard controller."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KeyboardCommand:
    """Planar velocity intent selected by one terminal key."""

    linear_mps: float
    angular_radps: float


ZERO_COMMAND = KeyboardCommand(0.0, 0.0)
DEFAULT_RELEASE_AWARE_KEYS = frozenset({"i", "u", "o"})


def command_for_key(
    key: str,
    *,
    linear_mps: float = 0.08,
    angular_radps: float = 0.40,
) -> KeyboardCommand:
    """Map one key to a bounded, latched command; every unknown key stops."""

    if not math.isfinite(linear_mps) or not 0.0 < linear_mps <= 0.10:
        raise ValueError("linear_mps must be finite and in (0, 0.10]")
    if not math.isfinite(angular_radps) or not 0.0 < angular_radps <= 0.50:
        raise ValueError("angular_radps must be finite and in (0, 0.50]")

    bindings = {
        "i": KeyboardCommand(linear_mps, 0.0),
        "u": KeyboardCommand(linear_mps, angular_radps),
        "o": KeyboardCommand(linear_mps, -angular_radps),
        "j": KeyboardCommand(0.0, angular_radps),
        "l": KeyboardCommand(0.0, -angular_radps),
        ",": KeyboardCommand(-linear_mps, 0.0),
        ".": KeyboardCommand(-linear_mps, angular_radps),
        "m": KeyboardCommand(-linear_mps, -angular_radps),
        "k": ZERO_COMMAND,
    }
    return bindings.get(key, ZERO_COMMAND)


def validate_allowed_motion_keys(keys: Iterable[str]) -> frozenset[str]:
    """Validate the deliberately small final-collection command alphabet."""

    normalized = frozenset(str(key).lower() for key in keys)
    if not normalized:
        raise ValueError("at least one motion key must be allowed")
    supported = frozenset({"i", "u", "o", "j", "l"})
    unsupported = normalized - supported
    if unsupported:
        raise ValueError(f"unsupported motion keys: {sorted(unsupported)}")
    return normalized


class ReleaseAwareKeyboard:
    """Track real key-down/key-up events and fail to zero on ambiguity.

    Repeated key-down events are idempotent. Pressing an unknown key, the stop
    key, or a second motion key clears the complete state so releasing one key
    cannot unexpectedly resume a previous command.
    """

    def __init__(
        self,
        *,
        allowed_motion_keys: Iterable[str] = DEFAULT_RELEASE_AWARE_KEYS,
        linear_mps: float = 0.08,
        angular_radps: float = 0.40,
    ) -> None:
        self.allowed_motion_keys = validate_allowed_motion_keys(allowed_motion_keys)
        # Validate both configured magnitudes before an event can select motion.
        command_for_key("i", linear_mps=linear_mps, angular_radps=angular_radps)
        self.linear_mps = float(linear_mps)
        self.angular_radps = float(angular_radps)
        self._pressed: set[str] = set()

    @property
    def pressed(self) -> frozenset[str]:
        return frozenset(self._pressed)

    @property
    def command(self) -> KeyboardCommand:
        if len(self._pressed) != 1:
            return ZERO_COMMAND
        key = next(iter(self._pressed))
        return command_for_key(
            key,
            linear_mps=self.linear_mps,
            angular_radps=self.angular_radps,
        )

    def key_down(self, key: str) -> KeyboardCommand:
        normalized = str(key).lower()
        if normalized == "k" or normalized not in self.allowed_motion_keys:
            return self.clear()
        if self._pressed and normalized not in self._pressed:
            return self.clear()
        self._pressed.add(normalized)
        return self.command

    def key_up(self, key: str) -> KeyboardCommand:
        self._pressed.discard(str(key).lower())
        return self.command

    def clear(self) -> KeyboardCommand:
        self._pressed.clear()
        return ZERO_COMMAND
