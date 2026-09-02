"""Can this machine actually back the neighbour table with 2 MB pages?

The training loop gathers random rows out of an 8.45 GB table forty times a
second. At 4 KB that table is 2.1 million page-table entries and the TLB covers
a vanishing fraction of it, so most gathers take a page-table walk; at 2 MB it
is 4,125 entries. Large-page allocations are also **non-pageable**, which would
stop the table being evicted to the pagefile under pressure -- we measured a
24 GB peak across the two pagefiles this morning.

Three things have to be true, and each is a separate failure:

  1. **SeLockMemoryPrivilege is granted.** Not held by default, not even by
     Administrators; it is granted through Local Security Policy and needs a
     logoff to take effect. `whoami /priv` listing it as Disabled means granted
     but not enabled in this token, which a process may do for itself.
  2. **It can be enabled** in the current process token.
  3. **A contiguous block is available.** Large pages need physically
     contiguous memory and Windows does not compact, so a request that would
     have succeeded at boot often fails on a machine that has been up for days.
     This is the one that usually bites, and it is why this probe exists rather
     than a try/except around the real allocation.

Nothing here is written to; the block is freed immediately.
"""

import argparse
import ctypes
import ctypes.wintypes as w
import sys

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
adv = ctypes.WinDLL("advapi32", use_last_error=True)

TOKEN_ADJUST_PRIVILEGES, TOKEN_QUERY = 0x0020, 0x0008
SE_PRIVILEGE_ENABLED = 0x0002
MEM_COMMIT, MEM_RESERVE, MEM_LARGE_PAGES, MEM_RELEASE = (0x1000, 0x2000,
                                                         0x20000000, 0x8000)
PAGE_READWRITE = 0x04


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", w.DWORD), ("HighPart", ctypes.c_long)]


class LUID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Luid", LUID), ("Attributes", w.DWORD)]


class TOKEN_PRIVILEGES(ctypes.Structure):
    _fields_ = [("PrivilegeCount", w.DWORD),
                ("Privileges", LUID_AND_ATTRIBUTES * 1)]


def enable_lock_memory():
    # GetCurrentProcess returns the pseudo-handle (HANDLE)-1; ctypes defaults
    # the return type to c_int, which truncates it on 64-bit and yields
    # ERROR_INVALID_HANDLE from OpenProcessToken.
    k32.GetCurrentProcess.restype = w.HANDLE
    k32.GetCurrentProcess.argtypes = []
    adv.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD,
                                     ctypes.POINTER(w.HANDLE)]
    tok = w.HANDLE()
    if not adv.OpenProcessToken(k32.GetCurrentProcess(),
                                TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                                ctypes.byref(tok)):
        return False, "OpenProcessToken failed ({})".format(
            ctypes.get_last_error())
    luid = LUID()
    if not adv.LookupPrivilegeValueW(None, "SeLockMemoryPrivilege",
                                     ctypes.byref(luid)):
        return False, "LookupPrivilegeValue failed ({})".format(
            ctypes.get_last_error())
    tp = TOKEN_PRIVILEGES(1, (LUID_AND_ATTRIBUTES * 1)(
        LUID_AND_ATTRIBUTES(luid, SE_PRIVILEGE_ENABLED)))
    ctypes.set_last_error(0)
    ok = adv.AdjustTokenPrivileges(tok, False, ctypes.byref(tp), 0, None, None)
    err = ctypes.get_last_error()
    if not ok:
        return False, "AdjustTokenPrivileges failed ({})".format(err)
    if err == 1300:      # ERROR_NOT_ALL_ASSIGNED
        return False, ("privilege not granted to this account -- grant "
                       "'Lock pages in memory' in secpol.msc and log off")
    return True, "enabled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=4.22,
                    help="block to attempt; 4.22 is the 768-d table, "
                         "8.45 the 1536-d one")
    a = ap.parse_args()

    if sys.platform != "win32":
        raise SystemExit("windows only")

    k32.GetLargePageMinimum.restype = ctypes.c_size_t
    lp = k32.GetLargePageMinimum()
    print("large page size      {:.0f} MB".format(lp / 1e6))
    if not lp:
        raise SystemExit("large pages not supported by this processor")

    ok, msg = enable_lock_memory()
    print("SeLockMemoryPrivilege {}".format(msg))
    if not ok:
        raise SystemExit(1)

    want = int(a.gb * 1e9)
    size = ((want + lp - 1) // lp) * lp
    print("requesting           {:.2f} GB in {:,} pages".format(
        size / 1e9, size // lp))

    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                 w.DWORD, w.DWORD]
    ctypes.set_last_error(0)
    p = k32.VirtualAlloc(None, size,
                         MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                         PAGE_READWRITE)
    if not p:
        err = ctypes.get_last_error()
        hint = ("  1450 = ERROR_NO_SYSTEM_RESOURCES: no contiguous physical "
                "block that large.\n  Large pages cannot be assembled from "
                "fragmented memory and Windows does not\n  compact, so this "
                "usually means the allocation must happen nearer to boot."
                if err == 1450 else "")
        print("FAILED, error {}".format(err))
        if hint:
            print(hint)
        raise SystemExit(1)

    print("SUCCESS -- allocated and locked; freeing")
    k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, w.DWORD]
    k32.VirtualFree(ctypes.c_void_p(p), 0, MEM_RELEASE)
    print("\nA block this size can be backed by 2 MB pages. Worth wiring into\n"
          "dataset.street_table, which is the only large private allocation\n"
          "in the training loop -- memory-mapped caches cannot use large pages\n"
          "at all, so they are unaffected either way.")


if __name__ == "__main__":
    main()
