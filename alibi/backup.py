"""Backup and restore functionality for Alibi database and vector index.

Creates and restores compressed archives with checksum verification.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
BACKUP_VERSION = "1.0"


@dataclass
class BackupManifest:
    """Manifest describing backup contents."""

    version: str = BACKUP_VERSION
    created_at: str = ""
    files: dict[str, str] = field(default_factory=dict)  # path -> checksum
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "files": self.files,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackupManifest:
        """Create from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            files=data.get("files", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BackupResult:
    """Result of a backup operation."""

    path: Path
    size_bytes: int
    file_count: int
    manifest: BackupManifest


@dataclass
class RestoreResult:
    """Result of a restore operation."""

    files_restored: int
    files_verified: int
    checksum_failures: list[str] = field(default_factory=list)


def _compute_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _add_directory_to_tar(
    tar: tarfile.TarFile,
    source_dir: Path,
    archive_prefix: str,
    manifest: BackupManifest,
) -> int:
    """Add directory contents to tar archive recursively.

    Returns number of files added.
    """
    if not source_dir.exists():
        logger.warning(f"Directory does not exist: {source_dir}")
        return 0

    count = 0
    for file_path in source_dir.rglob("*"):
        if file_path.is_file():
            # Compute relative path within archive
            rel_path = file_path.relative_to(source_dir)
            archive_path = f"{archive_prefix}/{rel_path}"

            # Add to archive
            tar.add(file_path, arcname=archive_path)

            # Compute and store checksum
            checksum = _compute_checksum(file_path)
            manifest.files[archive_path] = checksum
            count += 1

    return count


def create_backup(
    output_path: Path | str,
    db_path: Path | str,
    lance_path: Path | str | None = None,
    compression: str = "gz",
) -> BackupResult:
    """Create a backup archive.

    Args:
        output_path: Output tar.gz file path
        db_path: Path to SQLite database file
        lance_path: Optional path to LanceDB directory
        compression: Compression type (gz, bz2, xz, or empty for none)

    Returns:
        BackupResult with details

    Raises:
        ValueError: If no data to backup
        FileNotFoundError: If database file doesn't exist
    """
    output_path = Path(output_path)
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine compression mode
    if compression:
        mode = f"w:{compression}"
        if not str(output_path).endswith(f".tar.{compression}"):
            output_path = output_path.with_suffix(f".tar.{compression}")
    else:
        mode = "w"
        if output_path.suffix != ".tar":
            output_path = output_path.with_suffix(".tar")

    manifest = BackupManifest(
        created_at=datetime.now().isoformat(),
        metadata={
            "db_path": str(db_path),
            "lance_path": str(lance_path) if lance_path else None,
        },
    )

    total_files = 0

    with tarfile.open(output_path, mode) as tar:  # type: ignore[call-overload]
        # Add database file
        if db_path.exists():
            tar.add(db_path, arcname="database/alibi.db")
            manifest.files["database/alibi.db"] = _compute_checksum(db_path)
            total_files += 1
            logger.info(f"Added database file: {db_path}")

        # Add LanceDB directory if specified
        if lance_path:
            lance_dir = Path(lance_path)
            if lance_dir.exists():
                count = _add_directory_to_tar(tar, lance_dir, "lancedb", manifest)
                total_files += count
                logger.info(f"Added {count} files from LanceDB directory")

        # Write manifest
        manifest_json = json.dumps(manifest.to_dict(), indent=2)
        manifest_bytes = manifest_json.encode("utf-8")

        manifest_info = tarfile.TarInfo(name=MANIFEST_FILENAME)
        manifest_info.size = len(manifest_bytes)
        tar.addfile(manifest_info, io.BytesIO(manifest_bytes))

    if total_files == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("No data to backup - files are empty or don't exist")

    size_bytes = output_path.stat().st_size
    logger.info(
        f"Backup created: {output_path} ({size_bytes:,} bytes, {total_files} files)"
    )

    return BackupResult(
        path=output_path,
        size_bytes=size_bytes,
        file_count=total_files,
        manifest=manifest,
    )


def restore_backup(
    backup_path: Path | str,
    db_path: Path | str,
    lance_path: Path | str | None = None,
    verify_checksums: bool = True,
    overwrite: bool = False,
) -> RestoreResult:
    """Restore from a backup archive.

    Args:
        backup_path: Path to backup tar.gz file
        db_path: Target database file path
        lance_path: Optional target LanceDB directory
        verify_checksums: Verify file checksums after extraction
        overwrite: Overwrite existing files (default: False)

    Returns:
        RestoreResult with details

    Raises:
        FileNotFoundError: If backup file doesn't exist
        ValueError: If backup is invalid or checksum verification fails
    """
    backup_path = Path(backup_path)
    db_path = Path(db_path)

    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")

    # Check for existing data
    if not overwrite:
        if db_path.exists():
            raise ValueError(
                f"Database file exists: {db_path}. Use overwrite=True to proceed."
            )
        if lance_path:
            lance_dir = Path(lance_path)
            if lance_dir.exists() and any(lance_dir.iterdir()):
                raise ValueError(
                    f"LanceDB directory not empty: {lance_dir}. "
                    "Use overwrite=True to proceed."
                )

    # Extract to temp directory first
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract archive
        with tarfile.open(backup_path, "r:*") as tar:
            tar.extractall(temp_path, filter="data")

        # Read manifest
        manifest_path = temp_path / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise ValueError("Invalid backup: manifest.json not found")

        with open(manifest_path) as f:
            manifest = BackupManifest.from_dict(json.load(f))

        logger.info(f"Restoring backup from {manifest.created_at}")

        result = RestoreResult(files_restored=0, files_verified=0)

        # Verify checksums if requested
        if verify_checksums:
            for archive_path, expected_checksum in manifest.files.items():
                file_path = temp_path / archive_path
                if file_path.exists():
                    actual_checksum = _compute_checksum(file_path)
                    if actual_checksum == expected_checksum:
                        result.files_verified += 1
                    else:
                        result.checksum_failures.append(archive_path)
                        logger.warning(f"Checksum mismatch: {archive_path}")

            if result.checksum_failures:
                raise ValueError(
                    f"Checksum verification failed for {len(result.checksum_failures)} "
                    f"files: {', '.join(result.checksum_failures[:5])}"
                )

        # Clear existing data if overwrite
        if overwrite:
            if db_path.exists():
                db_path.unlink()
            if lance_path:
                lance_dir = Path(lance_path)
                if lance_dir.exists():
                    shutil.rmtree(lance_dir)

        # Restore database
        temp_db = temp_path / "database" / "alibi.db"
        if temp_db.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_db, db_path)
            result.files_restored += 1
            logger.info(f"Restored database to {db_path}")

        # Restore LanceDB directory
        temp_lance = temp_path / "lancedb"
        if lance_path and temp_lance.exists():
            lance_dir = Path(lance_path)
            lance_dir.mkdir(parents=True, exist_ok=True)
            for item in temp_lance.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(temp_lance)
                    dest = lance_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest)
                    result.files_restored += 1

    logger.info(
        f"Restore complete: {result.files_restored} files restored, "
        f"{result.files_verified} verified"
    )

    return result


def get_backup_info(backup_path: Path | str) -> BackupManifest:
    """Get information about a backup without extracting.

    Args:
        backup_path: Path to backup file

    Returns:
        BackupManifest with backup details

    Raises:
        FileNotFoundError: If backup doesn't exist
        ValueError: If backup is invalid
    """
    backup_path = Path(backup_path)

    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")

    with tarfile.open(backup_path, "r:*") as tar:
        try:
            manifest_file = tar.extractfile(MANIFEST_FILENAME)
            if manifest_file is None:
                raise ValueError("Invalid backup: manifest.json not found")
            manifest_data = json.load(manifest_file)
            return BackupManifest.from_dict(manifest_data)
        except KeyError:
            raise ValueError("Invalid backup: manifest.json not found")
