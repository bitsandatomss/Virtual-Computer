from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


class PatchRevertError(Exception):
    pass


class PatchManager:
    """Stage candidates beside the source and promote them atomically."""

    def __init__(self, source_path: str):
        self.source_path = Path(source_path).resolve()
        self.backup_path: str | None = None
        self._staged: set[Path] = set()
        self._promoted = False

    def backup(self) -> str:
        if self.backup_path and Path(self.backup_path).exists():
            return self.backup_path
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.source_path.name}.backup.",
            dir=self.source_path.parent,
        )
        os.close(descriptor)
        shutil.copy2(self.source_path, name)
        self.backup_path = name
        return name

    def stage(self, content: str) -> str:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.source_path.stem}.candidate.",
            suffix=self.source_path.suffix,
            dir=self.source_path.parent,
            text=True,
        )
        path = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            shutil.copymode(self.source_path, path)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        self._staged.add(path)
        return str(path)

    def promote(self, candidate_path: str) -> None:
        candidate = Path(candidate_path).resolve()
        if candidate not in self._staged:
            raise ValueError("Only a candidate staged by this manager can be promoted.")
        self.backup()
        try:
            os.replace(candidate, self.source_path)
            self._staged.discard(candidate)
            self._promoted = True
        except Exception:
            self.revert()
            raise

    def commit(self) -> None:
        """Forget the rollback copy after post-promotion verification succeeds."""
        self.cleanup()

    def apply_full_content(self, new_content: str) -> None:
        candidate = self.stage(new_content)
        self.backup()
        os.replace(candidate, self.source_path)
        self._staged.discard(Path(candidate))
        self._promoted = True

    def revert(self) -> None:
        if not self.backup_path or not Path(self.backup_path).exists():
            raise PatchRevertError("No backup file found to revert.")
        os.replace(self.backup_path, self.source_path)
        self._promoted = False

    def cleanup(self) -> None:
        for path in tuple(self._staged):
            path.unlink(missing_ok=True)
        self._staged.clear()
        if self.backup_path:
            Path(self.backup_path).unlink(missing_ok=True)
        self._promoted = False

    def __enter__(self) -> "PatchManager":
        return self

    def __exit__(self, *_args: object) -> None:
        try:
            if self._promoted and self.backup_path and Path(self.backup_path).exists():
                self.revert()
        finally:
            self.cleanup()
