"""Back a big array with Windows 2 MB pages, when the machine will allow it.

The neighbour table is gathered at random forty times a second. At 4 KB a
10.75 GB table is 2.6 million page-table entries and the TLB covers a vanishing
slice of it, so most gathers pay a page-table walk; at 2 MB it is 5,125 entries.
Large-page allocations are also **non-pageable**, so the table cannot be evicted
to the pagefile under pressure -- which is the failure that mattered, since a
private dirty page costs a pagefile *write* before eviction while a memmapped
page is clean and simply dropped.

**Whether it works depends on when you ask.** Large pages need physically
contiguous memory and Windows never compacts. Measured on this machine: right
after a reboot 10.75 GB succeeds; after days of uptime with 24 GB committed the
ceiling was 0.5 GB. So this is strictly opportunistic -- it returns None and the
caller keeps its existing behaviour.

Two things it deliberately does not do. It never frees: training processes are
short-lived and the OS reclaims at exit, so a free path would only add a way to
crash while torch still holds the buffer. And it refuses anything that would
lock more than `max_frac` of physical RAM, because locked pages come out of the
memory the dataloader workers and the page cache are already contending for.
"""

import ctypes
import sys

import numpy as np

_KEEP = []          # buffers must outlive the arrays torch wraps around them
_PRIV = None        # tri-state: None untried, True enabled, False unavailable


def _enable_privilege():
    """SeLockMemoryPrivilege, which large pages require. Granted through Local
    Security Policy, not held by default even by Administrators."""
    global _PRIV
    if _PRIV is not None:
        return _PRIV
    if sys.platform != "win32":
        _PRIV = False
        return False
    import ctypes.wintypes as w
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    adv = ctypes.WinDLL("advapi32", use_last_error=True)

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", w.DWORD), ("HighPart", ctypes.c_long)]

    class LAA(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", w.DWORD)]

    class TP(ctypes.Structure):
        _fields_ = [("PrivilegeCount", w.DWORD), ("Privileges", LAA * 1)]

    try:
        # the pseudo-handle is (HANDLE)-1; ctypes defaults to c_int and would
        # truncate it on 64-bit
        k32.GetCurrentProcess.restype = w.HANDLE
        adv.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD,
                                         ctypes.POINTER(w.HANDLE)]
        tok = w.HANDLE()
        if not adv.OpenProcessToken(k32.GetCurrentProcess(), 0x0020 | 0x0008,
                                    ctypes.byref(tok)):
            _PRIV = False
            return False
        luid = LUID()
        if not adv.LookupPrivilegeValueW(None, "SeLockMemoryPrivilege",
                                         ctypes.byref(luid)):
            _PRIV = False
            return False
        tp = TP(1, (LAA * 1)(LAA(luid, 0x0002)))
        ctypes.set_last_error(0)
        ok = adv.AdjustTokenPrivileges(tok, False, ctypes.byref(tp), 0,
                                       None, None)
        _PRIV = bool(ok) and ctypes.get_last_error() != 1300
    except Exception:
        _PRIV = False
    return _PRIV


def empty(shape, dtype=np.float16, max_frac=0.45, verbose=True):
    """A writable array on 2 MB pages, or None if the machine will not.

    `max_frac` caps how much of physical RAM may be locked. Locked pages are
    unavailable to everything else, and on a 31.7 GB machine already running
    eight dataloader workers, pinning most of it would trade one stall for
    another.
    """
    if sys.platform != "win32" or not _enable_privilege():
        return None
    import ctypes.wintypes as w
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetLargePageMinimum.restype = ctypes.c_size_t
    lp = k32.GetLargePageMinimum()
    if not lp:
        return None

    want = int(np.prod(shape)) * np.dtype(dtype).itemsize
    try:
        total = ctypes.c_ulonglong()
        k32.GetPhysicallyInstalledSystemMemory(ctypes.byref(total))
        phys = total.value * 1024
    except Exception:
        phys = 0
    if phys and want > max_frac * phys:
        if verbose:
            print("large pages declined: {:.2f} GB is over {:.0%} of {:.1f} GB "
                  "physical".format(want / 1e9, max_frac, phys / 1e9),
                  flush=True)
        return None

    size = ((want + lp - 1) // lp) * lp
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                 w.DWORD, w.DWORD]
    ctypes.set_last_error(0)
    p = k32.VirtualAlloc(None, size, 0x1000 | 0x2000 | 0x20000000, 0x04)
    if not p:
        if verbose:
            err = ctypes.get_last_error()
            print("large pages unavailable (error {}{}); using ordinary pages"
                  .format(err, ", memory too fragmented" if err == 1450
                          else ""), flush=True)
        return None

    buf = (ctypes.c_uint8 * size).from_address(p)
    _KEEP.append(buf)                       # never freed; the process exit is
    arr = np.frombuffer(buf, dtype=dtype, count=int(np.prod(shape)))
    if verbose:
        print("large pages: {:.2f} GB in {:,} x {:.0f} MB pages, locked"
              .format(size / 1e9, size // lp, lp / 1e6), flush=True)
    return arr.reshape(shape)
