"""Fail builds on expired security exceptions or unapproved scanner findings."""

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

IGNORE_FILE = Path(__file__).resolve().parents[1] / ".github/security-ignore.json"


def load_ignores(path, today):
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise TypeError("Security ignore file must contain a list.")

    ignores = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"tool", "id", "scope", "reason", "expires"}:
            raise ValueError("Each security exception requires tool, id, scope, reason and expires.")
        tool, finding_id, scope = entry["tool"], entry["id"], entry["scope"]
        if (
            not all(isinstance(entry[field], str) and entry[field].strip()
                    for field in ("tool", "id", "reason", "expires"))
            or tool not in SCANNERS
            or not isinstance(scope, list)
            or len(scope) < 2
            or (tool != "trivy" and len(scope) != 2)
            or not all(isinstance(value, str) and value.strip() for value in scope)
        ):
            raise ValueError("Invalid security exception metadata.")
        expiry = date.fromisoformat(entry["expires"])
        if expiry.isoformat() != entry["expires"]:
            raise ValueError("Use YYYY-MM-DD for security exception expiry.")
        if today > expiry:
            raise ValueError(f"Expired {tool} exception {finding_id} ({expiry})")
        target, *items = scope
        for item in items:
            key = (tool, finding_id, target, item)
            if key in ignores:
                raise ValueError(f"Duplicate {tool} exception {finding_id}")
            ignores.add(key)
    return ignores


def bandit_findings(report):
    if report["errors"]:
        raise ValueError("Bandit could not scan all application files.")
    for issue in report["results"]:
        code = "\n".join(
            line.partition(" ")[2].strip() for line in issue["code"].splitlines()
            if line.partition(" ")[0] == str(issue["line_number"])
        )
        yield "bandit", issue["test_id"], Path(issue["filename"]).as_posix(), code


def trivy_findings(report):
    if report.get("ArtifactType") != "container_image" or not report["Results"]:
        raise ValueError("Trivy produced no image scan targets.")
    for target in report["Results"]:
        for issue in target.get("Vulnerabilities") or []:
            package = f"{issue['PkgName']}@{issue['InstalledVersion']}"
            if fixed_version := issue.get("FixedVersion"):
                package += f" (fix available: {fixed_version})"
            yield "trivy", issue["VulnerabilityID"], target["Type"], package


def gitleaks_findings(report):
    if not isinstance(report, list):
        raise TypeError("Gitleaks report must contain a list of findings.")
    for issue in report:
        yield "gitleaks", issue["RuleID"], issue["File"], issue["Fingerprint"]


SCANNERS = {"bandit": bandit_findings, "trivy": trivy_findings, "gitleaks": gitleaks_findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=["expiry", *sorted(SCANNERS)])
    parser.add_argument("report", type=Path, nargs="?")
    args = parser.parse_args()

    try:
        ignores = load_ignores(IGNORE_FILE, datetime.now(timezone.utc).date())
        if args.tool == "expiry":
            print(f"Validated {len(ignores)} unexpired security exceptions.")
            return 0
        if args.report is None:
            parser.error("A JSON scanner report is required.")
        report = json.loads(args.report.read_text(encoding="utf-8"))
        findings = SCANNERS[args.tool](report)
        unapproved = 0
        for finding in findings:
            accepted = finding in ignores
            activity, finding_id, target, package = finding
            if activity == "trivy":
                target = f"{target}/{package}"
            disposition = "Accepted until expiry" if accepted else "Unapproved"
            print(f"{disposition} {activity} finding {finding_id} in {target}")
            unapproved += not accepted
        return int(unapproved > 0)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"Security gate failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
