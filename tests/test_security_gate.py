import json
from datetime import date

import pytest

from scripts.security_gate import bandit_findings, load_ignores, trivy_findings


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


def test_trivy_finding_includes_ecosystem_and_package():
    findings = trivy_findings({
        "ArtifactType": "container_image",
        "Results": [{"Type": "debian", "Vulnerabilities": [
            {"VulnerabilityID": "CVE-example", "PkgName": "libexample", "InstalledVersion": "1.0"},
        ]}],
    })
    assert list(findings) == [("trivy", "CVE-example", "debian", "libexample@1.0")]


def test_trivy_rejects_empty_image_scan():
    with pytest.raises(ValueError, match="no image scan targets"):
        list(trivy_findings({"ArtifactType": "container_image", "Results": []}))


def test_trivy_available_fix_requires_a_fresh_decision():
    issue = {"VulnerabilityID": "CVE-example", "PkgName": "libexample",
             "InstalledVersion": "1.0", "FixedVersion": "1.1"}
    report = {"ArtifactType": "container_image",
              "Results": [{"Type": "debian", "Vulnerabilities": [issue]}]}
    assert list(trivy_findings(report)) == [
        ("trivy", "CVE-example", "debian", "libexample@1.0 (fix available: 1.1)"),
    ]
