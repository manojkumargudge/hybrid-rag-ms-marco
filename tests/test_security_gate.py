import json
from datetime import date

import pytest

from scripts.security_gate import (
    bandit_findings,
    gitleaks_findings,
    load_ignores,
    trivy_findings,
)


@pytest.fixture
def ignore_file(tmp_path):
    path = tmp_path / "ignore.json"
    path.write_text(json.dumps([{
        "tool": "bandit",
        "id": "B301",
        "scope": ["app/example.py", "one = pickle.load(file)"],
        "reason": "Verified file in read-only storage.",
        "expires": "2026-11-30",
    }]))
    return path


def test_exception_expires_after_listed_utc_date(ignore_file):
    assert load_ignores(ignore_file, date(2026, 11, 30))
    with pytest.raises(ValueError, match="Expired bandit exception B301"):
        load_ignores(ignore_file, date(2026, 12, 1))


def test_bandit_exception_matches_a_single_statement(ignore_file):
    ignores = load_ignores(ignore_file, date(2026, 11, 30))
    report = {
        "errors": [],
        "results": [
            {"test_id": "B301", "filename": "./app/example.py", "line_number": line,
             "code": "10 one = pickle.load(file)\n11 two = pickle.load(file)\n"}
            for line in (10, 11)
        ],
    }
    allowed, denied = bandit_findings(report)
    assert allowed in ignores
    assert denied not in ignores


def test_history_exception_does_not_cover_another_commit(tmp_path):
    path = tmp_path / "ignore.json"
    fingerprint = f"{'a' * 40}:tests/example.py:generic-api-key:5"
    path.write_text(json.dumps([{
        "tool": "gitleaks", "id": "generic-api-key",
        "scope": ["tests/example.py", fingerprint],
        "reason": "Synthetic fixture in one historical commit.", "expires": "2027-01-31",
    }]))
    first = {"RuleID": "generic-api-key", "File": "tests/example.py", "Fingerprint": fingerprint}
    second = {**first, "Fingerprint": fingerprint.replace("a" * 40, "b" * 40)}
    known, new = gitleaks_findings([first, second])
    ignores = load_ignores(path, date(2026, 11, 30))
    assert known in ignores
    assert new not in ignores


def test_image_gate_keeps_high_and_critical_findings_without_available_fixes():
    base = {"PkgName": "example", "InstalledVersion": "1", "FixedVersion": ""}
    report = {"ArtifactType": "container_image", "Results": [{
        "Type": "debian",
        "Vulnerabilities": [
            {**base, "VulnerabilityID": severity, "Severity": severity}
            for severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        ],
    }]}
    assert list(trivy_findings(report)) == [
        ("trivy", "HIGH", "debian", "example@1"),
        ("trivy", "CRITICAL", "debian", "example@1"),
    ]


def test_image_gate_does_not_apply_unfixed_exception_when_fix_appears():
    issue = {
        "VulnerabilityID": "CVE-example", "Severity": "HIGH",
        "PkgName": "example", "InstalledVersion": "1", "FixedVersion": "2",
    }
    report = {"ArtifactType": "container_image", "Results": [{
        "Type": "debian", "Vulnerabilities": [issue],
    }]}
    assert list(trivy_findings(report)) == [
        ("trivy", "CVE-example", "debian", "example@1 (fix available: 2)"),
    ]


@pytest.mark.parametrize("report", [
    {"ArtifactType": "container_image", "Results": []},
    {"ArtifactType": "filesystem", "Results": [{"Type": "debian"}]},
])
def test_image_gate_rejects_missing_or_non_image_scan_targets(report):
    with pytest.raises(ValueError, match="no image scan targets"):
        list(trivy_findings(report))
