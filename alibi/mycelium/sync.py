"""Git sync detection and post-sync processing.

This module provides utilities for detecting git sync events
and triggering batch processing of new files.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from alibi.mycelium.watcher import MyceliumWatcher, DEFAULT_VAULT_PATH

logger = logging.getLogger(__name__)


@dataclass
class SyncStatus:
    """Status of a git sync operation."""

    success: bool
    new_files: list[Path]
    modified_files: list[Path]
    commit_hash: Optional[str] = None
    commit_message: Optional[str] = None
    error: Optional[str] = None


class SyncDetector:
    """Detects git sync events and tracks new files.

    This class monitors a git repository for sync events (pulls)
    and identifies new files that need processing.
    """

    def __init__(
        self,
        vault_path: Path | None = None,
        inbox_subpath: str = "inbox/documents",
    ) -> None:
        """Initialize the sync detector.

        Args:
            vault_path: Path to the git repository (vault)
            inbox_subpath: Subpath to monitor for new files
        """
        self.vault_path = vault_path or DEFAULT_VAULT_PATH
        self.inbox_path = self.vault_path / inbox_subpath
        self._last_commit: Optional[str] = None

    def _run_git(self, *args: str) -> tuple[bool, str]:
        """Run a git command in the vault directory.

        Args:
            *args: Git command arguments

        Returns:
            Tuple of (success, output)
        """
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.vault_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def get_current_commit(self) -> Optional[str]:
        """Get the current HEAD commit hash.

        Returns:
            Commit hash or None if not a git repo
        """
        success, output = self._run_git("rev-parse", "HEAD")
        return output if success else None

    def get_commit_message(self, commit_hash: str) -> Optional[str]:
        """Get the commit message for a given hash.

        Args:
            commit_hash: Git commit hash

        Returns:
            Commit message or None
        """
        success, output = self._run_git("log", "-1", "--format=%s", commit_hash)
        return output if success else None

    def check_for_sync(self) -> Optional[SyncStatus]:
        """Check if a sync (pull) has occurred since last check.

        Returns:
            SyncStatus if sync detected, None otherwise
        """
        current_commit = self.get_current_commit()
        if not current_commit:
            return SyncStatus(
                success=False,
                new_files=[],
                modified_files=[],
                error="Not a git repository",
            )

        if self._last_commit is None:
            # First check, just record the commit
            self._last_commit = current_commit
            return None

        if current_commit == self._last_commit:
            # No change
            return None

        # Detect new/modified files in inbox since last commit
        new_files, modified_files = self._get_changed_files(
            self._last_commit, current_commit
        )

        self._last_commit = current_commit

        return SyncStatus(
            success=True,
            new_files=new_files,
            modified_files=modified_files,
            commit_hash=current_commit,
            commit_message=self.get_commit_message(current_commit),
        )

    def _get_changed_files(
        self, old_commit: str, new_commit: str
    ) -> tuple[list[Path], list[Path]]:
        """Get files changed between two commits.

        Args:
            old_commit: Starting commit
            new_commit: Ending commit

        Returns:
            Tuple of (new_files, modified_files) in inbox
        """
        # Get diff with status (A=added, M=modified)
        success, output = self._run_git("diff", "--name-status", old_commit, new_commit)

        if not success:
            return [], []

        new_files: list[Path] = []
        modified_files: list[Path] = []
        inbox_rel = str(self.inbox_path.relative_to(self.vault_path))

        for line in output.split("\n"):
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue

            status, file_path = parts

            # Only process files in inbox
            if not file_path.startswith(inbox_rel):
                continue

            full_path = self.vault_path / file_path
            if status == "A":
                new_files.append(full_path)
            elif status == "M":
                modified_files.append(full_path)

        return new_files, modified_files

    def get_untracked_inbox_files(self) -> list[Path]:
        """Get untracked files in the inbox directory.

        Returns:
            List of untracked file paths
        """
        success, output = self._run_git(
            "ls-files", "--others", "--exclude-standard", str(self.inbox_path)
        )

        if not success:
            return []

        files: list[Path] = []
        for line in output.split("\n"):
            if line:
                files.append(self.vault_path / line)

        return files

    def pull_and_detect(self) -> SyncStatus:
        """Perform a git pull and detect new files.

        Returns:
            SyncStatus with sync results
        """
        old_commit = self.get_current_commit()

        # Perform git pull
        success, output = self._run_git("pull", "--ff-only")

        if not success:
            return SyncStatus(
                success=False,
                new_files=[],
                modified_files=[],
                error=f"Git pull failed: {output}",
            )

        new_commit = self.get_current_commit()

        if old_commit == new_commit:
            # No changes from pull
            return SyncStatus(
                success=True,
                new_files=[],
                modified_files=[],
                commit_hash=new_commit,
                commit_message="Already up to date",
            )

        # Get changed files
        new_files, modified_files = self._get_changed_files(
            old_commit or "", new_commit or ""
        )

        return SyncStatus(
            success=True,
            new_files=new_files,
            modified_files=modified_files,
            commit_hash=new_commit,
            commit_message=self.get_commit_message(new_commit or ""),
        )


def process_after_sync(
    vault_path: Path | None = None,
    generate_notes: bool = True,
    archive_processed: bool = False,
) -> SyncStatus:
    """Pull from git and process any new files.

    This is the main entry point for post-sync processing,
    suitable for use in a cron job or git hook.

    Args:
        vault_path: Path to the vault (defaults to ~/Obsidian/vault)
        generate_notes: Whether to generate Obsidian notes
        archive_processed: Whether to archive processed files

    Returns:
        SyncStatus with processing results
    """
    detector = SyncDetector(vault_path=vault_path)
    sync_status = detector.pull_and_detect()

    if not sync_status.success:
        logger.error(f"Sync failed: {sync_status.error}")
        return sync_status

    if not sync_status.new_files:
        logger.info("No new files to process")
        return sync_status

    logger.info(f"Found {len(sync_status.new_files)} new files to process")

    # Create watcher for processing
    watcher = MyceliumWatcher(
        vault_path=vault_path,
        generate_notes=generate_notes,
        archive_processed=archive_processed,
    )

    # Process each new file
    for file_path in sync_status.new_files:
        if file_path.exists() and file_path.is_file():
            logger.info(f"Processing: {file_path.name}")
            try:
                watcher._mycelium_handler = watcher._mycelium_handler or __import__(
                    "alibi.mycelium.watcher", fromlist=["MyceliumHandler"]
                ).MyceliumHandler(
                    vault_path=vault_path or DEFAULT_VAULT_PATH,
                    generate_notes=generate_notes,
                    archive_processed=archive_processed,
                )
                watcher._mycelium_handler.on_document_created(file_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")

    return sync_status


def create_post_pull_hook(vault_path: Path | None = None) -> str:
    """Generate content for a git post-merge hook.

    Args:
        vault_path: Path to the vault

    Returns:
        Shell script content for the hook
    """
    vault = vault_path or DEFAULT_VAULT_PATH

    return f"""#!/bin/bash
# Git post-merge hook for Mycelium vault
# Triggers Alibi processing after git pull

# Log file
LOG_FILE="$HOME/.alibi/post-sync.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "[$(date)] Post-merge hook triggered" >> "$LOG_FILE"

# Check if alibi CLI is available
if command -v lt &> /dev/null; then
    lt mycelium scan >> "$LOG_FILE" 2>&1
    echo "[$(date)] Processing complete" >> "$LOG_FILE"
else
    echo "[$(date)] Alibi CLI (lt) not found" >> "$LOG_FILE"
fi
"""


def install_post_pull_hook(vault_path: Path | None = None) -> Path:
    """Install a post-merge git hook in the vault.

    Args:
        vault_path: Path to the vault

    Returns:
        Path to the installed hook
    """
    vault = vault_path or DEFAULT_VAULT_PATH
    hooks_dir = vault / ".git" / "hooks"
    hook_path = hooks_dir / "post-merge"

    hooks_dir.mkdir(parents=True, exist_ok=True)

    hook_content = create_post_pull_hook(vault_path)
    hook_path.write_text(hook_content)
    hook_path.chmod(0o755)

    logger.info(f"Installed post-merge hook: {hook_path}")
    return hook_path
