import os
import pytest
from self_compiler.patch import PatchManager


def test_patch_lifecycle(tmp_path):
    # Setup origin
    src = tmp_path / "code.c"
    src.write_text("int main() { return 1; }")

    pm = PatchManager(str(src))

    # Apply patch
    pm.apply_full_content("int main() { return 0; }")

    assert src.read_text() == "int main() { return 0; }"
    assert os.path.exists(pm.backup_path)  # backup should exist

    # Revert patch
    pm.revert()
    assert src.read_text() == "int main() { return 1; }"
    assert not os.path.exists(pm.backup_path)  # backup should be cleaned up


def test_staging_does_not_touch_source_until_promotion(tmp_path):
    src = tmp_path / "code.c"
    src.write_text("original", encoding="utf-8")
    pm = PatchManager(str(src))

    staged = pm.stage("candidate")
    assert src.read_text(encoding="utf-8") == "original"

    pm.promote(staged)
    assert src.read_text(encoding="utf-8") == "candidate"
    assert os.path.exists(pm.backup_path)
    pm.revert()
    assert src.read_text(encoding="utf-8") == "original"


def test_cleanup_removes_staged_and_backup_files(tmp_path):
    src = tmp_path / "code.c"
    src.write_text("original", encoding="utf-8")
    pm = PatchManager(str(src))
    staged = pm.stage("candidate")
    backup = pm.backup()
    pm.cleanup()
    assert not os.path.exists(staged)
    assert not os.path.exists(backup)


def test_context_rolls_back_uncommitted_promotion(tmp_path):
    src = tmp_path / "code.c"
    src.write_text("original", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with PatchManager(str(src)) as pm:
            staged = pm.stage("candidate")
            pm.promote(staged)
            raise RuntimeError("post-promotion verification crashed")
    assert src.read_text(encoding="utf-8") == "original"
