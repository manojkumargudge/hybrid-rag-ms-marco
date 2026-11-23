import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest
import yaml

from scripts import publish_image

SHA = "ab" * 20
DIGEST = f"{publish_image.REGISTRY_REPOSITORY}@sha256:" + "cd" * 32
WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/release-image.yml"
LOCAL_ID = "sha256:" + "ef" * 32
IDENTITY = f"{LOCAL_ID} https://github.com/{publish_image.REPOSITORY} {SHA} 1.0.0"


@pytest.fixture
def release_env():
    return {
        "GITHUB_SHA": SHA,
        "REQUESTED_SHA": SHA,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REPOSITORY": publish_image.REPOSITORY,
        "GITHUB_ACTOR": "test-operator",
        "PUBLISH_REQUESTED": "true",
        "GHCR_TOKEN": "offline-test-token",
    }


def test_release_workflow_keeps_registry_write_out_of_pr_runs():
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"pull_request", "workflow_dispatch"}
    assert workflow["on"]["workflow_dispatch"]["inputs"]["publish"]["default"] == "false"
    assert workflow["on"]["workflow_dispatch"]["inputs"]["commit_sha"]["required"] == "true"
    assert workflow["permissions"] == {"contents": "read"}
    verify, publish = (workflow["jobs"][job] for job in ("verify", "publish"))
    assert "permissions" not in verify
    assert all("GHCR_TOKEN" not in step.get("env", {}) for step in verify["steps"])
    assert "github.event_name == 'workflow_dispatch'" in publish["if"]
    assert "github.ref == 'refs/heads/main'" in publish["if"]
    assert "&& inputs.publish" in publish["if"]
    assert publish["permissions"] == {"contents": "read", "packages": "write"}
    assert publish["environment"] == "ghcr-release"
    enable, build, push = [
        step for step in publish["steps"]
        if step.get("name") in {
            "Require publication to be enabled",
            "Build and validate the release candidate",
            "Publish the checked image under its commit SHA",
        }
    ]
    assert enable["env"] == {"GHCR_RELEASE_ENABLED": "${{ secrets.GHCR_RELEASE_ENABLED }}"}
    assert enable["run"] == 'test "$GHCR_RELEASE_ENABLED" = true'
    assert build["run"] == "bash scripts/check_release_image.sh"
    assert verify["steps"][-1]["run"] == build["run"]
    assert verify["steps"][-1]["env"] == build["env"] == {"REQUESTED_SHA": "${{ inputs.commit_sha }}"}
    assert "GHCR_TOKEN" in push["env"]
    gates = workflow["jobs"]["release-gates"]
    assert publish["needs"] == "release-gates"
    assert gates["if"] == publish["if"]
    assert "inputs.commit_sha == github.sha" in gates["if"]
    assert gates["uses"] == "./.github/workflows/ci.yml"
    assert "secrets" not in gates
    assert "permissions" not in gates
    ci = yaml.load(WORKFLOW.with_name("ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert "workflow_call" in ci["on"]
    assert {"test", "security", "docker"} <= ci["jobs"].keys()
    for job in (verify, publish):
        for step in job["steps"]:
            if "uses" in step:
                pin = step["uses"].split("@")[-1]
                assert len(pin) == 40
                int(pin, 16)


@pytest.mark.parametrize(("name", "invalid"), [
    ("GITHUB_EVENT_NAME", "pull_request"),
    ("GITHUB_REF", "refs/heads/feat/operations"),
    ("REQUESTED_SHA", "ff" * 20),
    ("GITHUB_REPOSITORY", "other/fork"),
    ("PUBLISH_REQUESTED", "false"),
])
def test_publication_refuses_wrong_context_before_docker(release_env, name, invalid):
    release_env[name] = invalid
    with pytest.raises(ValueError, match="requested main revision"):
        publish_image.publish(release_env)


@pytest.mark.parametrize(("status", "message", "allowed", "recovered"), [
    (0, "", False, True),
    (1, "denied: permission refused", False, True),
    (1, "no such manifest: expected tag", True, True),
    (1, "no such manifest: expected tag", True, False),
])
def test_publisher_pushes_only_when_the_commit_tag_is_absent(
    monkeypatch, release_env, status, message, allowed, recovered,
):
    calls = []
    target = f"{publish_image.REGISTRY_REPOSITORY}:sha-{SHA}"

    def fake_docker(*args, env, token=None, check=True):
        assert "GHCR_TOKEN" not in env
        assert "offline-test-token" not in args
        assert token == ("offline-test-token" if args[0] == "login" else None)
        calls.append(args)
        if args[:2] == ("image", "inspect"):
            inspected = args[-1]
            if inspected == target:
                result = json.dumps([DIGEST])
            elif inspected == DIGEST and not recovered:
                result = IDENTITY.replace(LOCAL_ID, "sha256:" + "00" * 32)
            else:
                result = IDENTITY
            return CompletedProcess(args, 0, result, "")
        if args == ("manifest", "inspect", target):
            assert not check
            return CompletedProcess(args, status, "", message)
        return CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(publish_image, "docker", fake_docker)
    if allowed and recovered:
        assert publish_image.publish(release_env) == DIGEST
        assert ("push", target) in calls
        assert ("pull", DIGEST) in calls
        assert calls[-1][-1] == DIGEST
    elif allowed:
        with pytest.raises(ValueError, match="pulled by digest"):
            publish_image.publish(release_env)
    else:
        with pytest.raises(ValueError, match="already exists|Refusing to publish"):
            publish_image.publish(release_env)
        assert not any(args[0] in ("tag", "push") for args in calls)


def test_publisher_rejects_wrong_version_before_login(monkeypatch, release_env):
    def inspect(*args, **kwargs):
        assert args[:2] == ("image", "inspect")
        return CompletedProcess(args, 0, IDENTITY.replace("1.0.0", "2.0.0"), "")

    monkeypatch.setattr(publish_image, "docker", inspect)
    with pytest.raises(ValueError, match="OCI labels"):
        publish_image.publish(release_env)
