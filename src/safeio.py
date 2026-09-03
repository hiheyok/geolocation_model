"""Write a file completely, or not at all.

Every artifact this project produces is later read back by something that
assumes it is whole: a checkpoint by `load_model`, a state file by the runner on
restart, a report by whoever is reading the numbers. A write interrupted part
way leaves a file that exists, has a plausible size, and is wrong -- which is
this project's most expensive failure shape, and the reason `done.u8.npy` and
the k-NN provenance checks exist.

Interruption is not hypothetical here. Runs are unattended and overnight, a
training rung writes a 24 MB checkpoint at the end of every epoch, and the
runner is killed and relaunched by design. `os.replace` is atomic on the same
volume on both Windows and POSIX, so writing beside the target and renaming
turns a torn write into no write at all.

The temp file is deliberately created in the *destination directory* rather
than the system temp: `os.replace` is only atomic within a filesystem, and
checkpoints live on C: while some caches live on E:.
"""

import os
from pathlib import Path


def _tmp_for(path):
    path = Path(path)
    return path.with_name(path.name + ".tmp{}".format(os.getpid()))


def replace_from(tmp, path):
    """Rename tmp onto path, cleaning up tmp if the rename fails."""
    try:
        os.replace(tmp, path)
    except OSError:
        try:
            Path(tmp).unlink()
        except OSError:
            pass
        raise


def save_torch(obj, path, **kw):
    """torch.save, atomically. Returns the path written."""
    import torch

    path = Path(path)
    tmp = _tmp_for(path)
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f, **kw)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    replace_from(tmp, path)
    return path


def write_text(path, text, encoding="utf-8"):
    """Path.write_text, atomically, and never in the platform's default codec.

    The encoding is not optional in practice: this runs on Windows, where the
    default is cp1252 and a single non-ASCII character in a report raises or
    mojibakes. That has already happened once here.
    """
    path = Path(path)
    tmp = _tmp_for(path)
    try:
        with open(tmp, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    replace_from(tmp, path)
    return path
