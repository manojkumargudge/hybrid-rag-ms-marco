def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 text decoded as Latin-1 without changing valid text."""

    repaired = text
    for _ in range(2):
        try:
            candidate = repaired.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            break
        if candidate == repaired:
            break
        repaired = candidate
    return repaired