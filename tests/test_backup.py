"""Tests for backup and restore functionality."""

import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from alibi.backup import (
    BackupManifest,
    BackupResult,
    RestoreResult,
    create_backup,
    get_backup_info,
    restore_backup,
)


@pytest.fixture
def test_db_file(tmp_path: Path) -> Path:
    """Create a test database file with some content."""
    db_path = tmp_path / "test_alibi.db"
    db_path.write_text("Test database content\nLine 2\nLine 3")
    return db_path


@pytest.fixture
def test_lance_dir(tmp_path: Path) -> Path:
    """Create a test LanceDB directory with some files."""
    lance_dir = tmp_path / "lancedb"
    lance_dir.mkdir()

    # Create some test files in the lance directory
    (lance_dir / "data.lance").write_text("Vector data 1")
    (lance_dir / "metadata.json").write_text('{"version": "1.0"}')

    # Create a subdirectory with files
    subdir = lance_dir / "subdir"
    subdir.mkdir()
    (subdir / "file1.txt").write_text("Subdir file 1")
    (subdir / "file2.txt").write_text("Subdir file 2")

    return lance_dir


class TestCreateBackup:
    """Tests for create_backup function."""

    def test_create_backup_database_only(
        self, tmp_path: Path, test_db_file: Path
    ) -> None:
        """Create backup with just the database."""
        backup_path = tmp_path / "backup.tar.gz"

        result = create_backup(
            output_path=backup_path,
            db_path=test_db_file,
        )

        assert isinstance(result, BackupResult)
        assert result.path == backup_path
        assert result.path.exists()
        assert result.size_bytes > 0
        assert result.file_count == 1
        assert "database/alibi.db" in result.manifest.files

    def test_create_backup_with_vectors(
        self, tmp_path: Path, test_db_file: Path, test_lance_dir: Path
    ) -> None:
        """Create backup including LanceDB directory."""
        backup_path = tmp_path / "backup.tar.gz"

        result = create_backup(
            output_path=backup_path,
            db_path=test_db_file,
            lance_path=test_lance_dir,
        )

        assert isinstance(result, BackupResult)
        assert result.path == backup_path
        assert result.path.exists()
        assert result.size_bytes > 0
        # Should have 1 db file + 4 lance files (2 top-level + 2 in subdir)
        assert result.file_count == 5
        assert "database/alibi.db" in result.manifest.files
        assert "lancedb/data.lance" in result.manifest.files
        assert "lancedb/metadata.json" in result.manifest.files
        assert "lancedb/subdir/file1.txt" in result.manifest.files
        assert "lancedb/subdir/file2.txt" in result.manifest.files

    def test_backup_creates_manifest(self, tmp_path: Path, test_db_file: Path) -> None:
        """Verify manifest.json is created with correct structure."""
        backup_path = tmp_path / "backup.tar.gz"

        result = create_backup(
            output_path=backup_path,
            db_path=test_db_file,
        )

        # Extract and verify manifest
        with tarfile.open(backup_path, "r:gz") as tar:
            manifest_file = tar.extractfile("manifest.json")
            assert manifest_file is not None
            manifest_data = json.load(manifest_file)

            assert "version" in manifest_data
            assert manifest_data["version"] == "1.0"
            assert "created_at" in manifest_data
            assert "files" in manifest_data
            assert "metadata" in manifest_data
            assert manifest_data["metadata"]["db_path"] == str(test_db_file)
            assert manifest_data["metadata"]["lance_path"] is None

        # Verify manifest in result
        assert result.manifest.version == "1.0"
        assert result.manifest.created_at != ""
        assert len(result.manifest.files) == 1

    def test_backup_checksum_verification(
        self, tmp_path: Path, test_db_file: Path
    ) -> None:
        """Verify checksums are computed correctly."""
        backup_path = tmp_path / "backup.tar.gz"

        result = create_backup(
            output_path=backup_path,
            db_path=test_db_file,
        )

        # Get checksum from manifest
        db_checksum = result.manifest.files["database/alibi.db"]
        assert len(db_checksum) == 64  # SHA256 hex digest length

        # Verify it's a valid hex string
        int(db_checksum, 16)  # Should not raise ValueError

    def test_create_backup_missing_db(self, tmp_path: Path) -> None:
        """Test error when database doesn't exist."""
        backup_path = tmp_path / "backup.tar.gz"
        nonexistent_db = tmp_path / "nonexistent.db"

        with pytest.raises(FileNotFoundError, match="Database file not found"):
            create_backup(
                output_path=backup_path,
                db_path=nonexistent_db,
            )

    def test_create_backup_with_compression_types(
        self, tmp_path: Path, test_db_file: Path
    ) -> None:
        """Test different compression types."""
        # Test gzip (default)
        backup_gz = tmp_path / "backup.tar.gz"
        result_gz = create_backup(backup_gz, test_db_file, compression="gz")
        assert result_gz.path.suffix == ".gz"
        assert result_gz.path.exists()

        # Test bzip2
        backup_bz2 = tmp_path / "backup.tar.bz2"
        result_bz2 = create_backup(backup_bz2, test_db_file, compression="bz2")
        assert result_bz2.path.suffix == ".bz2"
        assert result_bz2.path.exists()

        # Test xz
        backup_xz = tmp_path / "backup.tar.xz"
        result_xz = create_backup(backup_xz, test_db_file, compression="xz")
        assert result_xz.path.suffix == ".xz"
        assert result_xz.path.exists()

        # Test no compression
        backup_tar = tmp_path / "backup.tar"
        result_tar = create_backup(backup_tar, test_db_file, compression="")
        assert result_tar.path.suffix == ".tar"
        assert result_tar.path.exists()


class TestRestoreBackup:
    """Tests for restore_backup function."""

    def test_restore_backup_success(self, tmp_path: Path, test_db_file: Path) -> None:
        """Restore from backup and verify files."""
        # Create backup
        backup_path = tmp_path / "backup.tar.gz"
        create_backup(backup_path, test_db_file)

        # Restore to new location
        restore_db = tmp_path / "restored" / "alibi.db"

        result = restore_backup(
            backup_path=backup_path,
            db_path=restore_db,
        )

        assert isinstance(result, RestoreResult)
        assert result.files_restored == 1
        assert result.files_verified == 1
        assert len(result.checksum_failures) == 0
        assert restore_db.exists()
        assert restore_db.read_text() == test_db_file.read_text()

    def test_restore_backup_with_vectors(
        self, tmp_path: Path, test_db_file: Path, test_lance_dir: Path
    ) -> None:
        """Restore backup including LanceDB directory."""
        # Create backup
        backup_path = tmp_path / "backup.tar.gz"
        create_backup(backup_path, test_db_file, test_lance_dir)

        # Restore to new location
        restore_db = tmp_path / "restored" / "alibi.db"
        restore_lance = tmp_path / "restored" / "lancedb"

        result = restore_backup(
            backup_path=backup_path,
            db_path=restore_db,
            lance_path=restore_lance,
        )

        assert result.files_restored == 5  # 1 db + 4 lance files
        assert result.files_verified == 5
        assert len(result.checksum_failures) == 0

        # Verify database
        assert restore_db.exists()
        assert restore_db.read_text() == test_db_file.read_text()

        # Verify lance files
        assert (restore_lance / "data.lance").exists()
        assert (restore_lance / "metadata.json").exists()
        assert (restore_lance / "subdir" / "file1.txt").exists()
        assert (restore_lance / "subdir" / "file2.txt").exists()

        # Verify content
        assert (restore_lance / "data.lance").read_text() == "Vector data 1"
        assert (restore_lance / "metadata.json").read_text() == '{"version": "1.0"}'

    def test_restore_backup_checksum_failure(
        self, tmp_path: Path, test_db_file: Path
    ) -> None:
        """Test that checksum failures are detected."""
        # Create backup
        backup_path = tmp_path / "backup.tar.gz"
        create_backup(backup_path, test_db_file)

        # Tamper with the backup by extracting, modifying, and repacking
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        with tarfile.open(backup_path, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")

        # Modify the database file
        db_file = extract_dir / "database" / "alibi.db"
        db_file.write_text("TAMPERED CONTENT")

        # Repack
        tampered_backup = tmp_path / "tampered.tar.gz"
        with tarfile.open(tampered_backup, "w:gz") as tar:
            for item in extract_dir.rglob("*"):
                if item.is_file():
                    arcname = str(item.relative_to(extract_dir))
                    tar.add(item, arcname=arcname)

        # Try to restore - should fail checksum verification
        restore_db = tmp_path / "restored" / "alibi.db"

        with pytest.raises(ValueError, match="Checksum verification failed"):
            restore_backup(
                backup_path=tampered_backup,
                db_path=restore_db,
                verify_checksums=True,
            )

    def test_restore_backup_without_verification(
        self, tmp_path: Path, test_db_file: Path
    ) -> None:
        """Test restore with checksum verification disabled."""
        # Create backup
        backup_path = tmp_path / "backup.tar.gz"
        create_backup(backup_path, test_db_file)

        # Restore without verification
        restore_db = tmp_path / "restored" / "alibi.db"

        result = restore_backup(
            backup_path=backup_path,
            db_path=restore_db,
            verify_checksums=False,
        )

        assert result.files_restored == 1
        assert result.files_verified == 0  # No verification performed
        assert len(result.checksum_failures) == 0
        assert restore_db.exists()

    def test_restore_backup_existing_data(
        self, tmp_path: Path, test_db_file: Path
    ) -> None:
        """Test error when overwrite=False and data exists."""
        # Create backup
        backup_path = tmp_path / "backup.tar.gz"
        create_backup(backup_path, test_db_file)

        # Create existing database
        restore_db = tmp_path / "restored" / "alibi.db"
        restore_db.parent.mkdir(parents=True)
        restore_db.write_text("Existing data")

        # Should fail without overwrite
        with pytest.raises(
            ValueError, match="Database file exists.*Use overwrite=True"
        ):
            restore_backup(
                backup_path=backup_path,
                db_path=restore_db,
                overwrite=False,
            )

    def test_restore_backup_existing_lance_data(
        self, tmp_path: Path, test_db_file: Path, test_lance_dir: Path
    ) -> None:
        """Test error when overwrite=False and LanceDB directory exists."""
        # Create backup
        backup_path = tmp_path / "backup.tar.gz"
        create_backup(backup_path, test_db_file, test_lance_dir)

        # Create existing lance directory with content
        restore_db = tmp_path / "restored" / "alibi.db"
        restore_lance = tmp_path / "restored" / "lancedb"
        restore_lance.mkdir(parents=True)
        (restore_lance / "existing.txt").write_text("Existing")

        # Should fail without overwrite
        with pytest.raises(
            ValueError, match="LanceDB directory not empty.*Use overwrite=True"
        ):
            restore_backup(
                backup_path=backup_path,
                db_path=restore_db,
                lance_path=restore_lance,
                overwrite=False,
            )

    def test_restore_backup_with_overwrite(
        self, tmp_path: Path, test_db_file: Path
    ) -> None:
        """Test restore with overwrite=True replaces existing data."""
        # Create backup
        backup_path = tmp_path / "backup.tar.gz"
        create_backup(backup_path, test_db_file)

        # Create existing database
        restore_db = tmp_path / "restored" / "alibi.db"
        restore_db.parent.mkdir(parents=True)
        restore_db.write_text("Existing data")

        # Should succeed with overwrite
        result = restore_backup(
            backup_path=backup_path,
            db_path=restore_db,
            overwrite=True,
        )

        assert result.files_restored == 1
        assert restore_db.read_text() == test_db_file.read_text()

    def test_restore_backup_missing_file(self, tmp_path: Path) -> None:
        """Test error when backup file doesn't exist."""
        nonexistent_backup = tmp_path / "nonexistent.tar.gz"
        restore_db = tmp_path / "restored" / "alibi.db"

        with pytest.raises(FileNotFoundError, match="Backup file not found"):
            restore_backup(
                backup_path=nonexistent_backup,
                db_path=restore_db,
            )

    def test_restore_backup_invalid_archive(self, tmp_path: Path) -> None:
        """Test error when backup is not a valid tar archive."""
        invalid_backup = tmp_path / "invalid.tar.gz"
        invalid_backup.write_text("Not a tar file")

        restore_db = tmp_path / "restored" / "alibi.db"

        with pytest.raises(Exception):  # tarfile.ReadError or similar
            restore_backup(
                backup_path=invalid_backup,
                db_path=restore_db,
            )

    def test_restore_backup_missing_manifest(
        self, tmp_path: Path, test_db_file: Path
    ) -> None:
        """Test error when backup is missing manifest.json."""
        # Create a tar without manifest
        backup_path = tmp_path / "no_manifest.tar.gz"

        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(test_db_file, arcname="database/alibi.db")

        restore_db = tmp_path / "restored" / "alibi.db"

        with pytest.raises(ValueError, match="Invalid backup: manifest.json not found"):
            restore_backup(
                backup_path=backup_path,
                db_path=restore_db,
            )


class TestGetBackupInfo:
    """Tests for get_backup_info function."""

    def test_get_backup_info(
        self, tmp_path: Path, test_db_file: Path, test_lance_dir: Path
    ) -> None:
        """Get backup info without extracting."""
        # Create backup
        backup_path = tmp_path / "backup.tar.gz"
        original_result = create_backup(backup_path, test_db_file, test_lance_dir)

        # Get info
        manifest = get_backup_info(backup_path)

        assert isinstance(manifest, BackupManifest)
        assert manifest.version == "1.0"
        assert manifest.created_at == original_result.manifest.created_at
        assert len(manifest.files) == 5  # 1 db + 4 lance files
        assert manifest.metadata["db_path"] == str(test_db_file)
        assert manifest.metadata["lance_path"] == str(test_lance_dir)

    def test_get_backup_info_database_only(
        self, tmp_path: Path, test_db_file: Path
    ) -> None:
        """Get backup info for database-only backup."""
        backup_path = tmp_path / "backup.tar.gz"
        create_backup(backup_path, test_db_file)

        manifest = get_backup_info(backup_path)

        assert isinstance(manifest, BackupManifest)
        assert len(manifest.files) == 1
        assert "database/alibi.db" in manifest.files
        assert manifest.metadata["lance_path"] is None

    def test_get_backup_info_missing_file(self, tmp_path: Path) -> None:
        """Test error when backup file doesn't exist."""
        nonexistent_backup = tmp_path / "nonexistent.tar.gz"

        with pytest.raises(FileNotFoundError, match="Backup file not found"):
            get_backup_info(nonexistent_backup)

    def test_get_backup_info_invalid_backup(self, tmp_path: Path) -> None:
        """Test error when backup is missing manifest."""
        # Create a tar without manifest
        backup_path = tmp_path / "no_manifest.tar.gz"
        dummy_file = tmp_path / "dummy.txt"
        dummy_file.write_text("dummy")

        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(dummy_file, arcname="dummy.txt")

        with pytest.raises(ValueError, match="Invalid backup: manifest.json not found"):
            get_backup_info(backup_path)


class TestBackupManifest:
    """Tests for BackupManifest class."""

    def test_manifest_to_dict(self) -> None:
        """Test BackupManifest.to_dict() conversion."""
        manifest = BackupManifest(
            version="1.0",
            created_at="2024-01-01T12:00:00",
            files={"file1.txt": "abc123", "file2.txt": "def456"},
            metadata={"key": "value"},
        )

        data = manifest.to_dict()

        assert data["version"] == "1.0"
        assert data["created_at"] == "2024-01-01T12:00:00"
        assert data["files"] == {"file1.txt": "abc123", "file2.txt": "def456"}
        assert data["metadata"] == {"key": "value"}

    def test_manifest_from_dict(self) -> None:
        """Test BackupManifest.from_dict() creation."""
        data = {
            "version": "1.0",
            "created_at": "2024-01-01T12:00:00",
            "files": {"file1.txt": "abc123"},
            "metadata": {"key": "value"},
        }

        manifest = BackupManifest.from_dict(data)

        assert manifest.version == "1.0"
        assert manifest.created_at == "2024-01-01T12:00:00"
        assert manifest.files == {"file1.txt": "abc123"}
        assert manifest.metadata == {"key": "value"}

    def test_manifest_from_dict_defaults(self) -> None:
        """Test BackupManifest.from_dict() with missing fields uses defaults."""
        data: dict[str, Any] = {}

        manifest = BackupManifest.from_dict(data)

        assert manifest.version == "1.0"
        assert manifest.created_at == ""
        assert manifest.files == {}
        assert manifest.metadata == {}

    def test_manifest_roundtrip(self) -> None:
        """Test BackupManifest to_dict() -> from_dict() roundtrip."""
        original = BackupManifest(
            version="1.0",
            created_at="2024-01-01T12:00:00",
            files={"file1.txt": "abc123", "file2.txt": "def456"},
            metadata={"db_path": "/path/to/db", "lance_path": "/path/to/lance"},
        )

        data = original.to_dict()
        restored = BackupManifest.from_dict(data)

        assert restored.version == original.version
        assert restored.created_at == original.created_at
        assert restored.files == original.files
        assert restored.metadata == original.metadata
