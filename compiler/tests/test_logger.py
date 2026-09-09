import io

from rich.console import Console

from self_compiler import logger


def test_status_logging_is_safe_on_legacy_windows_encoding(monkeypatch):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(logger, "console", Console(file=stream, force_terminal=False))

    logger.print_step("working")
    logger.print_success("done")
    logger.print_error("failed")
    stream.flush()

    assert raw.getvalue().decode("cp1252").splitlines() == [
        "> working",
        "OK done",
        "ERROR failed",
    ]
