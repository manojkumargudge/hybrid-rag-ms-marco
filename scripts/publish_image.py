"""Publish the already-smoked image on an explicitly authorized main dispatch."""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

REPOSITORY = "manojkumargudge/hybrid-rag-ms-marco"
REGISTRY_REPOSITORY = f"ghcr.io/{REPOSITORY}"
LOCAL_IMAGE = "hybrid-rag:release-check"


def authorized_revision(env: dict[str, str]) -> str:
    revision = env.get("GITHUB_SHA", "")
    if (
        env.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or env.get("GITHUB_REF") != "refs/heads/main"
        or env.get("GITHUB_REPOSITORY") != REPOSITORY
        or env.get("PUBLISH_REQUESTED") != "true"
        or env.get("REQUESTED_SHA") != revision
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise ValueError("Publication requires an exact, explicitly requested main revision.")
    return revision


def docker(*args: str, env: dict[str, str], token: str | None = None, check: bool = True):
    return subprocess.run(
        ["docker", *args], env=env, input=token, text=True, capture_output=True, check=check,
    )


def publish(env: dict[str, str]) -> str:
    revision = authorized_revision(env)
    cli_env = dict(env)
    token = cli_env.pop("GHCR_TOKEN")
    if not token or not cli_env.get("GITHUB_ACTOR"):
        raise ValueError("Publication credentials are missing.")
    tag = f"{REGISTRY_REPOSITORY}:sha-{revision}"
    with tempfile.TemporaryDirectory(prefix="rag-ghcr-") as directory:
        cli_env["DOCKER_CONFIG"] = directory
        local = docker(
            "image", "inspect", "--format",
            '{{index .Config.Labels "org.opencontainers.image.revision"}}',
            LOCAL_IMAGE, env=cli_env,
        ).stdout.strip()
        if local != revision:
            raise ValueError("The tested local image revision does not match the requested SHA.")
        docker("login", "ghcr.io", "--username", cli_env["GITHUB_ACTOR"], "--password-stdin",
               env=cli_env, token=token)
        remote = docker("manifest", "inspect", tag, env=cli_env, check=False)
        if remote.returncode == 0:
            raise ValueError("The immutable commit tag already exists.")
        message = remote.stderr.strip().lower()
        if not (message == "manifest unknown"
                or message.startswith(("no such manifest:", "manifest unknown:"))):
            raise ValueError("Could not confirm tag absence. Refusing to publish.")
        docker("tag", LOCAL_IMAGE, tag, env=cli_env)
        docker("push", tag, env=cli_env)
        values = json.loads(docker(
            "image", "inspect", "--format", "{{json .RepoDigests}}", tag, env=cli_env,
        ).stdout)
        digests = [ref for ref in values or []
                   if re.fullmatch(rf"{re.escape(REGISTRY_REPOSITORY)}@sha256:[0-9a-f]{{64}}", ref)]
        if len(digests) != 1:
            raise ValueError("Push returned no unique registry digest. Inspect GHCR before retrying.")
        docker("manifest", "inspect", digests[0], env=cli_env)
        return digests[0]


def main() -> int:
    try:
        digest = publish(dict(os.environ))
    except subprocess.CalledProcessError as error:
        print(f"Docker command failed with exit {error.returncode}. Check the registry before retrying.")
        return 1
    except (ValueError, KeyError) as error:
        print(f"Publication stopped: {error}")
        return 1
    print(f"Published and verified {digest}")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(f"Published digest: `{digest}`\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
