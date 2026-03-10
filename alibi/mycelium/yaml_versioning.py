"""Git auto-commit for .alibi.yaml files in the vault.

Best-effort: failures are logged, never block processing.
Batch commits by default (not per-file) to reduce git overhead.

Config:
    ALIBI_YAML_GIT_VERSIONING: Enable/disable (default True)
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from alibi.mycelium.watcher import DEFAULT_VAULT_PATH

logger = logging.getLogger(__name__)


def _is_enabled() -> bool:
    return os.environ.get("ALIBI_YAML_GIT_VERSIONING", "true").lower() in (
        "true",
        "1",
        "yes",
    )


class YamlVersioner:
    """Git auto-commit for .alibi.yaml files in the vault."""

    def __init__(self, vault_path: Path | None = None):
        self._vault_path = vault_path or DEFAULT_VAULT_PATH
        self._pending: list[Path] = []

    def _run_git(self, *args: str) -> tuple[bool, str]:
        """Run a git command in the vault directory."""
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self._vault_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def _is_git_repo(self) -> bool:
        ok, _ = self._run_git("rev-parse", "--git-dir")
        return ok

    def track(self, yaml_path: Path) -> None:
        """Mark a YAML for next commit batch.

        Validates path is within vault. Skips if versioning is disabled.
        """
        if not _is_enabled():
            return

        try:
            yaml_path.resolve().relative_to(self._vault_path.resolve())
        except ValueError:
            logger.debug(f"YAML path {yaml_path} is outside vault, skipping git track")
            return

        self._pending.append(yaml_path)

    def commit_pending(self) -> bool:
        """git add + git commit all tracked files. Returns success."""
        if not self._pending:
            return True

        if not _is_enabled():
            self._pending.clear()
            return True

        if not self._is_git_repo():
            logger.debug("Vault is not a git repo, skipping YAML versioning")
            self._pending.clear()
            return False

        files = list(self._pending)
        self._pending.clear()

        # Stage files
        for f in files:
            ok, err = self._run_git("add", str(f))
            if not ok:
                logger.warning(f"git add failed for {f}: {err}")

        # Check if there's anything staged
        ok, diff = self._run_git("diff", "--cached", "--name-only")
        if not diff.strip():
            return True  # Nothing to commit

        file_list = ", ".join(f.name for f in files)
        msg = f"alibi: update extraction cache ({len(files)} files)\n\n{file_list}"

        ok, err = self._run_git("commit", "-m", msg)
        if ok:
            logger.info(f"Git committed {len(files)} YAML file(s)")
        else:
            logger.warning(f"git commit failed: {err}")
        return ok

    def commit_single(self, yaml_path: Path) -> bool:
        """Immediately commit one file (for user corrections)."""
        if not _is_enabled():
            return True

        if not self._is_git_repo():
            return False

        try:
            yaml_path.resolve().relative_to(self._vault_path.resolve())
        except ValueError:
            return False

        ok, err = self._run_git("add", str(yaml_path))
        if not ok:
            logger.warning(f"git add failed for {yaml_path}: {err}")
            return False

        ok, diff = self._run_git("diff", "--cached", "--name-only")
        if not diff.strip():
            return True

        msg = f"alibi: update extraction cache\n\n{yaml_path.name}"
        ok, err = self._run_git("commit", "-m", msg)
        if ok:
            logger.info(f"Git committed correction: {yaml_path.name}")
        else:
            logger.warning(f"git commit failed: {err}")
        return ok


# Module singleton
_versioner: YamlVersioner | None = None


def get_yaml_versioner(vault_path: Path | None = None) -> YamlVersioner:
    """Get or create the module-level YamlVersioner singleton."""
    global _versioner
    if _versioner is None:
        _versioner = YamlVersioner(vault_path)
    return _versioner


def reset_yaml_versioner() -> None:
    """Reset the module-level singleton (for tests)."""
    global _versioner
    _versioner = None
