"""Keep Windows from throttling the demo process. Standard library only; a no-op off Windows.

What it prevents. Windows applies "power throttling" (EcoQoS) to the processes of an app that is
in the background or minimised, most aggressively on battery: their threads go to the efficiency
cores at a low clock speed. A live demo sits in exactly that state, because the terminal that
started it is behind the browser. Measured on the development laptop on 10 Oct 2026 with the
quickstart, on battery: about 800 ms per detector call instead of about 30, and 27 s end to end
instead of 2 s.

What it does. A process may tell Windows not to throttle it: SetProcessInformation with
ProcessPowerThrottling, the EXECUTION_SPEED bit set in ControlMask and clear in StateMask (what
the Windows documentation calls HighQoS). That is one attribute of this one process. It changes no
power plan and no system setting, and it ends with the process.

    keep_full_speed()   call once at start-up; returns what happened, in words; never raises
    state()             "opted out" | "throttled" | "left to Windows" | "unknown"

Set P7_DEMO_ALLOW_THROTTLING=1 to skip the opt-out (that is how its effect is measured).
"""
from __future__ import annotations

import os
import sys

_EXECUTION_SPEED = 0x1               # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
_PROCESS_POWER_THROTTLING = 4        # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_status: str | None = None


def _kernel32():
    import ctypes
    from ctypes import wintypes as wt

    class State(ctypes.Structure):   # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG), ("StateMask", wt.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wt.HANDLE
    for fn in (k32.SetProcessInformation, k32.GetProcessInformation):
        fn.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        fn.restype = wt.BOOL
    return ctypes, k32, State


def _set(throttled: bool) -> int:
    """Take control of the EXECUTION_SPEED bit and set it (throttled) or clear it (never throttle).
    Returns 0, or the Windows error code. `throttled=True` exists for the tests, which reproduce
    the state Windows puts a background process in."""
    ctypes, k32, State = _kernel32()
    s = State(1, _EXECUTION_SPEED, _EXECUTION_SPEED if throttled else 0)
    ok = k32.SetProcessInformation(k32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
                                   ctypes.byref(s), ctypes.sizeof(s))
    return 0 if ok else (ctypes.get_last_error() or -1)


def state() -> str:
    """What this process has asked of Windows. "left to Windows" means no request was made, so
    Windows decides (and throttles a background process on battery)."""
    if sys.platform != "win32":
        return "unknown"
    try:
        ctypes, k32, State = _kernel32()
        s = State(1, 0, 0)
        if not k32.GetProcessInformation(k32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
                                         ctypes.byref(s), ctypes.sizeof(s)):
            return "unknown"
    except (OSError, AttributeError):
        return "unknown"
    if not s.ControlMask & _EXECUTION_SPEED:
        return "left to Windows"
    return "throttled" if s.StateMask & _EXECUTION_SPEED else "opted out"


def _opt_out() -> str:
    if sys.platform != "win32":
        return "not needed: throttling of background processes is a Windows feature"
    if os.environ.get("P7_DEMO_ALLOW_THROTTLING"):
        return "skipped: P7_DEMO_ALLOW_THROTTLING is set, so Windows may throttle this process"
    try:
        err = _set(throttled=False)
    except (OSError, AttributeError) as e:          # a Windows build without the call
        return f"unavailable: {type(e).__name__}: {e}"
    if err:
        return f"unavailable: Windows refused the request (error {err})"
    return "opted out: Windows will not slow this process down in the background or on battery"


def keep_full_speed() -> str:
    """Opt this process out of Windows power throttling, once. Returns what happened."""
    global _status
    if _status is None:
        _status = _opt_out()
    return _status
