"""Structural checks for citations to the passages returned with an answer."""

import re

INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
_CLAIM = re.compile(r"([^\[\]\r\n]+?)((?:[ \t]+\[pid:[0-9]+\])+)")
_PASSAGE_ID = re.compile(r"\[pid:([0-9]+)\]")
_SENTENCE_BREAK = re.compile(r"""[.!?]["'”’)]*\s+""")


def has_returned_citations(answer: str, passages: list[dict]) -> bool:
    allowed = {str(passage["passage_id"]) for passage in passages}
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        return False
    for line in lines:
        claim = _CLAIM.fullmatch(line)
        if (
            claim is None
            or not claim[1].strip()
            or _SENTENCE_BREAK.search(claim[1].strip())
            or not set(_PASSAGE_ID.findall(claim[2])) <= allowed
        ):
            return False
    return True
