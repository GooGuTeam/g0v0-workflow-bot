from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from aiowhales import AsyncDockerClient
from aiowhales.exceptions import ContainerNotFound
from nonebot.log import logger

DOCKER_IMAGE = "agent-workflow-image"
CONTAINER_NAME = "agent-workflow-{owner}-{repo}"
BUILD_CONTEXT = str(Path(__file__).resolve().parents[3])
WORKSPACE_PID_DIR = "/run/pydantic-ai"


class RepoContainerLockedError(Exception):
    def __init__(self, key: str, pr: int) -> None:
        super().__init__(f"Container for {key} is locked for PR #{pr}")


class DockerError(Exception):
    pass


class PullRequestContainer:
    def __init__(self, owner: str, repo: str) -> None:
        self.owner = owner
        self.repo = repo
        self.container_name = CONTAINER_NAME.format(owner=owner, repo=repo)
        self.working_dir = f"/workspace/{owner}-{repo}"
        self.container_id: str | None = None
        self.pr: int | None = None

    async def create(self) -> None:
        logger.info("Looking up Docker container: name={}", self.container_name)
        async with AsyncDockerClient() as docker:
            containers = await docker.containers.list(all=True)
            matched = next((c for c in containers if c.name.lstrip("/") == self.container_name), None)
            if matched is not None:
                self.container_id = matched.id
                logger.info(
                    "Reusing Docker container: name={} id={} status={}", self.container_name, matched.id, matched.status
                )
                if matched.status != "running":
                    logger.info("Starting existing Docker container: name={}", self.container_name)
                    await docker.containers.start(matched.id)
                return

            images = await docker.images.list()
            if not any(tag in {DOCKER_IMAGE, f"{DOCKER_IMAGE}:latest"} for img in images for tag in (img.tags or [])):
                logger.info("Building Docker image: image={} context={}", DOCKER_IMAGE, BUILD_CONTEXT)
                async for _ in docker.images.build(
                    BUILD_CONTEXT,
                    dockerfile="Dockerfile",
                    tags=[DOCKER_IMAGE],
                ):
                    pass
                logger.info("Docker image built: image={}", DOCKER_IMAGE)
            logger.info("Creating Docker container: name={} image={}", self.container_name, DOCKER_IMAGE)
            container = await docker.containers.run(
                DOCKER_IMAGE,
                ["sleep", "infinity"],
                name=self.container_name,
                detach=True,
                labels={"ai.pydantic.workspace": "true"},
            )
            self.container_id = container.id
            logger.info("Docker container started: name={} id={}", self.container_name, self.container_id)

    async def prepare_workspace(self) -> None:
        """Prepare DockerSandbox's timeout/cancellation state directory, including in older containers."""
        assert self.container_id is not None
        async with AsyncDockerClient() as docker:
            result = await docker.exec.run(self.container_id, ["test", "-w", WORKSPACE_PID_DIR])
            if result.exit_code != 0:
                logger.info(
                    "Preparing DockerSandbox PID directory: container={} path={}", self.container_id, WORKSPACE_PID_DIR
                )
                # aiowhales ExecAPI has no user override; the sandbox already requires the Docker CLI.
                process = await asyncio.create_subprocess_exec(
                    "docker",
                    "exec",
                    "--user",
                    "root",
                    self.container_id,
                    "install",
                    "-d",
                    "-m",
                    "1777",
                    WORKSPACE_PID_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await process.communicate()
                if process.returncode != 0:
                    raise DockerError(f"Failed to prepare DockerSandbox PID directory: {stderr.decode().strip()}")
                result = await docker.exec.run(self.container_id, ["test", "-w", WORKSPACE_PID_DIR])
                if result.exit_code != 0:
                    raise DockerError("DockerSandbox PID directory is not writable by the container's default user")
        logger.info("DockerSandbox workspace permissions ready: container={}", self.container_id)

    async def init(self, pr: int, *, head_sha: str | None = None) -> None:
        if self.pr is not None:
            raise RepoContainerLockedError(f"{self.owner}/{self.repo}", self.pr)
        self.pr = pr
        logger.info("Assigning PR to container: repo={}/{} pr={}", self.owner, self.repo, pr)
        await self.create()
        await self.prepare_workspace()
        assert self.container_id is not None
        async with AsyncDockerClient() as docker:
            result = await docker.exec.run(self.container_id, ["test", "-d", f"{self.working_dir}/.git"])
            if result.exit_code != 0:
                logger.info("Cloning repository: repo={}/{} workdir={}", self.owner, self.repo, self.working_dir)
                result = await docker.exec.run(
                    self.container_id,
                    [
                        "git",
                        "clone",
                        "--recursive",
                        f"https://github.com/{self.owner}/{self.repo}.git",
                        self.working_dir,
                    ],
                )
                if result.exit_code != 0:
                    raise DockerError(f"Failed to clone {self.owner}/{self.repo}")

            # Fetch into a remote ref so an already checked-out PR can be refreshed after a force push.
            ref = f"refs/remotes/origin/review-{pr}"
            for command in (
                ["git", "reset", "--hard"],
                ["git", "clean", "-fd"],
                ["git", "fetch", "--force", "origin", f"pull/{pr}/head:{ref}"],
                ["git", "switch", "-C", f"pr-{pr}", head_sha or ref],
                ["git", "submodule", "update", "--init", "--recursive"],
            ):
                logger.info(
                    "Preparing PR checkout: repo={}/{} pr={} command={}", self.owner, self.repo, pr, " ".join(command)
                )
                result = await docker.exec.run(self.container_id, command, workdir=self.working_dir)
                if result.exit_code != 0:
                    raise DockerError(f"Failed to prepare {self.owner}/{self.repo} PR #{pr}: {' '.join(command)}")

    async def request_external_repo(self, remote: str, branch: str | None = None, pr: int | None = None) -> str:
        """Shallow-clone an external dependency into its own directory in this container."""
        if not branch and pr is None:
            raise ValueError("At least one of branch or pr must be supplied")
        if pr is not None and pr <= 0:
            raise ValueError("pr must be a positive integer")
        url = urlsplit(remote)
        if url.scheme != "https" or not url.hostname or not url.path.strip("/") or url.username or url.password:
            raise ValueError("remote must be an HTTPS repository URL without embedded credentials")
        if branch is not None and (not branch or branch.startswith("-")):
            raise ValueError("branch must be a nonempty Git branch name")
        if self.container_id is None:
            raise DockerError("The review container is not available")

        workdir = f"/workspace/external-{uuid4().hex}"
        clone = ["git", "clone", "--depth", "1"]
        if branch is not None:
            clone.extend(["-b", branch])
        clone.extend(["--", remote, workdir])
        logger.info("Cloning external review dependency: container={} branch={} pr={}", self.container_id, branch, pr)
        async with AsyncDockerClient() as docker:
            result = await docker.exec.run(self.container_id, clone)
            if result.exit_code != 0:
                raise DockerError("Failed to shallow-clone the external repository")
            commands = []
            if pr is not None:
                ref = f"refs/remotes/origin/review-{pr}"
                commands.extend(
                    [
                        ["git", "fetch", "--depth", "1", "--force", "origin", f"pull/{pr}/head:{ref}"],
                        ["git", "switch", "-C", f"pr-{pr}", ref],
                    ]
                )
            commands.append(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"])
            for command in commands:
                result = await docker.exec.run(self.container_id, command, workdir=workdir)
                if result.exit_code != 0:
                    raise DockerError(f"Failed to prepare external repository: {' '.join(command)}")
        logger.info("External review dependency ready: container={} path={} pr={}", self.container_id, workdir, pr)
        return workdir

    def free_job(self) -> None:
        logger.info("Releasing PR assignment: name={} pr={}", self.container_name, self.pr)
        self.pr = None

    async def free_container(self) -> None:
        if self.pr is not None:
            raise RuntimeError("Cannot free container while a PR is still assigned")
        async with AsyncDockerClient() as docker:
            # Also find containers created just before an interrupted initialization.
            containers = await docker.containers.list(all=True)
            matched = next((c for c in containers if c.name.lstrip("/") == self.container_name), None)
            if matched is not None:
                logger.info("Removing Docker container: name={} id={}", self.container_name, matched.id)
                with suppress(ContainerNotFound):
                    await docker.containers.remove(matched.id, force=True)
        self.container_id = None


import asyncio
