import signal

import pytest

from tabbench_bio.predictions import UnitTimeoutError, _time_budget

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"), reason="POSIX alarm signals are required"
)


def test_time_budget_converts_native_keyboard_interrupt_after_alarm():
    with pytest.raises(UnitTimeoutError, match="exceeded the 60s"):
        with _time_budget(60):
            try:
                signal.raise_signal(signal.SIGALRM)
            except UnitTimeoutError:
                raise KeyboardInterrupt


def test_time_budget_preserves_genuine_keyboard_interrupt():
    with pytest.raises(KeyboardInterrupt):
        with _time_budget(60):
            raise KeyboardInterrupt
