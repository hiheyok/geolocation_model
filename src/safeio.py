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


def file_stamp(path, missing="absent"):
    """Identify a file by its bytes-on-disk, cheaply: size and mtime.

    Names are not identity. A cache keyed on the *path* of its PCA basis or
    its image root hands back vectors built from a different basis the moment
    that file is rebuilt under the same name -- and rebuilding a basis under
    the same name is the normal way this project produces one. The same
    mistake, keyed on a tag rather than a path, served the previous model's
    errors under the new model's name.

    Size and mtime catch a rewrite and cost a stat. Hashing hundreds of MB on
    every cache lookup would not, and the failure being guarded here is an
    accidental rebuild, not an adversary.
    """
    try:
        st = Path(path).stat()
    except OSError:
        return missing
    return "{:x}{:x}".format(st.st_size, st.st_mtime_ns)[-12:]


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
