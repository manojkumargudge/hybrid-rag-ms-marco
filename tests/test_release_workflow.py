import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest
import yaml

from scripts import publish_image

SHA = "ab" * 20
DIGEST = f"{publish_image.REGISTRY_REPOSITORY}@sha256:" + "cd" * 32
WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/release-image.yml"


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
    assert workflow["permissions"] == {"contents": "read"}
    verify, publish = (workflow["jobs"][job] for job in ("verify", "publish"))
    assert "permissions" not in verify
    assert all("GHCR_TOKEN" not in step.get("env", {}) for step in verify["steps"])
    assert "github.event_name == 'workflow_dispatch'" in publish["if"]
    assert "github.ref == 'refs/heads/main'" in publish["if"]
    assert "&& inputs.publish" in publish["if"]
    assert publish["permissions"] == {"contents": "read", "packages": "write"}
    assert publish["environment"] == "ghcr-release"
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


@pytest.mark.parametrize(("status", "message", "allowed"), [
    (0, "", False),
    (1, "denied: permission refused", False),
    (1, "no such manifest: expected tag", True),
])
def test_publisher_pushes_only_when_the_commit_tag_is_absent(
    monkeypatch, release_env, status, message, allowed,
):
    calls = []
    target = f"{publish_image.REGISTRY_REPOSITORY}:sha-{SHA}"

    def fake_docker(*args, env, token=None, check=True):
        assert "GHCR_TOKEN" not in env
        assert "offline-test-token" not in args
        assert token == ("offline-test-token" if args[0] == "login" else None)
        calls.append(args)
        if args[:2] == ("image", "inspect"):
            result = SHA if args[-1] == publish_image.LOCAL_IMAGE else json.dumps([DIGEST])
            return CompletedProcess(args, 0, result, "")
        if args == ("manifest", "inspect", target):
            assert not check
            return CompletedProcess(args, status, "", message)
        return CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(publish_image, "docker", fake_docker)
    if allowed:
        assert publish_image.publish(release_env) == DIGEST
        assert ("push", target) in calls
        assert calls[-1] == ("manifest", "inspect", DIGEST)
    else:
        with pytest.raises(ValueError, match="already exists|Refusing to publish"):
            publish_image.publish(release_env)
        assert not any(args[0] in ("tag", "push") for args in calls)
