"""Tests for alibi.services.annotation."""

from unittest.mock import MagicMock, patch

import pytest

from alibi.services import annotation as svc

MODULE = "alibi.services.annotation"


@pytest.fixture
def db() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------


def test_annotate_returns_id(db: MagicMock) -> None:
    expected_id = "ann-uuid-1234"
    with patch(f"{MODULE}._add_annotation", return_value=expected_id) as mock:
        result = svc.annotate(
            db,
            target_type="fact",
            target_id="fact-001",
            annotation_type="person",
            key="bought_for",
            value="Maria",
        )

    assert result == expected_id
    mock.assert_called_once_with(
        db,
        annotation_type="person",
        target_type="fact",
        target_id="fact-001",
        key="bought_for",
        value="Maria",
        metadata=None,
        source="user",
    )


def test_annotate_passes_metadata_and_source(db: MagicMock) -> None:
    with patch(f"{MODULE}._add_annotation", return_value="id-x") as mock:
        svc.annotate(
            db,
            target_type="vendor",
            target_id="v-001",
            annotation_type="project",
            key="project",
            value="Renovation",
            metadata={"split": 0.5},
            source="auto",
        )

    _, kwargs = mock.call_args
    assert kwargs["metadata"] == {"split": 0.5}
    assert kwargs["source"] == "auto"


# ---------------------------------------------------------------------------
# get_annotations
# ---------------------------------------------------------------------------


def test_get_annotations_returns_list(db: MagicMock) -> None:
    rows = [{"id": "a1", "key": "bought_for", "value": "Maria"}]
    with patch(f"{MODULE}._get_annotations", return_value=rows) as mock:
        result = svc.get_annotations(db, target_type="fact", target_id="fact-001")

    assert result == rows
    mock.assert_called_once_with(
        db,
        target_type="fact",
        target_id="fact-001",
        annotation_type=None,
    )


def test_get_annotations_passes_annotation_type_filter(db: MagicMock) -> None:
    with patch(f"{MODULE}._get_annotations", return_value=[]) as mock:
        svc.get_annotations(
            db,
            target_type="identity",
            target_id="id-001",
            annotation_type="project",
        )

    _, kwargs = mock.call_args
    assert kwargs["annotation_type"] == "project"


def test_get_annotations_empty(db: MagicMock) -> None:
    with patch(f"{MODULE}._get_annotations", return_value=[]):
        result = svc.get_annotations(db, target_type="fact", target_id="missing")

    assert result == []


# ---------------------------------------------------------------------------
# update_annotation
# ---------------------------------------------------------------------------


def test_update_annotation_returns_true_on_success(db: MagicMock) -> None:
    with patch(f"{MODULE}._update_annotation", return_value=True) as mock:
        result = svc.update_annotation(db, "ann-001", value="Bob")

    assert result is True
    mock.assert_called_once_with(
        db,
        annotation_id="ann-001",
        value="Bob",
        metadata=None,
    )


def test_update_annotation_returns_false_when_not_found(db: MagicMock) -> None:
    with patch(f"{MODULE}._update_annotation", return_value=False):
        result = svc.update_annotation(db, "nonexistent", value="x")

    assert result is False


def test_update_annotation_passes_metadata(db: MagicMock) -> None:
    with patch(f"{MODULE}._update_annotation", return_value=True) as mock:
        svc.update_annotation(db, "ann-002", metadata={"split": 0.3})

    _, kwargs = mock.call_args
    assert kwargs["metadata"] == {"split": 0.3}
    assert kwargs["value"] is None


# ---------------------------------------------------------------------------
# delete_annotation
# ---------------------------------------------------------------------------


def test_delete_annotation_returns_true_on_success(db: MagicMock) -> None:
    with patch(f"{MODULE}._delete_annotation", return_value=True) as mock:
        result = svc.delete_annotation(db, "ann-001")

    assert result is True
    mock.assert_called_once_with(db, "ann-001")


def test_delete_annotation_returns_false_when_not_found(db: MagicMock) -> None:
    with patch(f"{MODULE}._delete_annotation", return_value=False):
        result = svc.delete_annotation(db, "missing-id")

    assert result is False
