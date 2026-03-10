"""Fact correction — user-driven reassignment of bundles between clouds.

When cloud formation makes a mistake (e.g., a card slip from store A gets
matched to a receipt from store B because of similar amounts), the user
needs to:

1. Inspect the fact to see which bundles/atoms are inside
2. Move the wrongly-assigned bundle to the correct cloud (or a new one)
3. Re-collapse both the source and target clouds into facts

This module provides the orchestration layer that coordinates v2_store
mutations with cloud re-collapse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from alibi.clouds.collapse import try_collapse
from alibi.db.connection import DatabaseManager
from alibi.db.models import Cloud, CloudStatus
from alibi.db import v2_store

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Result of a fact correction operation."""

    success: bool = False
    error: str | None = None
    source_cloud_id: str | None = None
    target_cloud_id: str | None = None
    source_fact_id: str | None = None
    target_fact_id: str | None = None
    deleted_clouds: int = 0


def move_bundle(
    db: DatabaseManager,
    bundle_id: str,
    target_cloud_id: str | None = None,
) -> CorrectionResult:
    """Move a bundle to a different cloud and re-collapse both clouds.

    Args:
        db: Database manager.
        bundle_id: The bundle to move.
        target_cloud_id: Cloud to move it to, or None to create a new cloud.

    Returns:
        CorrectionResult with new fact IDs after re-collapse.
    """
    result = CorrectionResult()

    # 1. Find the source cloud
    source_cloud_id = v2_store.get_cloud_for_bundle(db, bundle_id)
    if not source_cloud_id:
        result.error = f"Bundle {bundle_id} is not assigned to any cloud"
        return result
    result.source_cloud_id = source_cloud_id

    # 2. Delete the source fact (if collapsed)
    source_fact = v2_store.get_fact_for_cloud(db, source_cloud_id)
    if source_fact:
        v2_store.delete_fact(db, source_fact["id"])

    # 3. Move the bundle
    if target_cloud_id:
        # Delete target fact too (will re-collapse)
        target_fact = v2_store.get_fact_for_cloud(db, target_cloud_id)
        if target_fact:
            v2_store.delete_fact(db, target_fact["id"])

        ok = v2_store.move_bundle_to_cloud(db, bundle_id, target_cloud_id)
        if not ok:
            result.error = f"Failed to move bundle to cloud {target_cloud_id}"
            return result
        result.target_cloud_id = target_cloud_id
    else:
        new_cloud_id = v2_store.move_bundle_to_new_cloud(db, bundle_id)
        if not new_cloud_id:
            result.error = "Failed to create new cloud for bundle"
            return result
        result.target_cloud_id = new_cloud_id

    # 4. Re-collapse source cloud (if it still has bundles)
    source_bundles = v2_store.get_bundles_in_cloud(db, source_cloud_id)
    if source_bundles:
        source_fact_id = _try_recollapse(db, source_cloud_id)
        result.source_fact_id = source_fact_id
    else:
        # Source cloud is now empty — clean it up
        result.deleted_clouds = v2_store.delete_empty_clouds(db)

    # 5. Re-collapse target cloud
    target_cid = result.target_cloud_id
    if target_cid:
        target_fact_id = _try_recollapse(db, target_cid)
        result.target_fact_id = target_fact_id

    # 6. Identity feedback: teach the registry when vendors are merged
    try:
        _teach_identity_from_correction(db, bundle_id, result.target_cloud_id)
    except Exception as e:
        logger.debug(f"Identity feedback skipped: {e}")

    # 7. Record correction for cloud formation learning (fail-safe)
    try:
        _record_correction_feature(
            db,
            bundle_id,
            result.source_cloud_id,
            result.target_cloud_id,
            target_cloud_id is None,  # was_false_positive = splitting out
        )
    except Exception as e:
        logger.debug(f"Correction recording skipped: {e}")

    result.success = True
    return result


def recollapse_cloud(
    db: DatabaseManager,
    cloud_id: str,
) -> str | None:
    """Force re-collapse of a cloud into a fact.

    Deletes existing fact (if any) and re-runs collapse logic.

    Returns:
        The new fact ID if collapse succeeded, None if the cloud
        stays in FORMING status.
    """
    # Delete existing fact
    existing = v2_store.get_fact_for_cloud(db, cloud_id)
    if existing:
        v2_store.delete_fact(db, existing["id"])

    return _try_recollapse(db, cloud_id)


def mark_disputed(db: DatabaseManager, cloud_id: str) -> bool:
    """Mark a cloud as disputed (needs human review).

    Deletes the existing fact and sets cloud status to DISPUTED.
    """
    existing = v2_store.get_fact_for_cloud(db, cloud_id)
    if existing:
        v2_store.delete_fact(db, existing["id"])

    v2_store.set_cloud_status(db, cloud_id, CloudStatus.DISPUTED.value)
    return True


def _record_correction_feature(
    db: DatabaseManager,
    bundle_id: str,
    source_cloud_id: str | None,
    target_cloud_id: str | None,
    was_false_positive: bool,
) -> None:
    """Extract features from the correction and record for learning."""
    from alibi.clouds.learning import CorrectionFeatureVector, record_correction
    from alibi.clouds.formation import (
        extract_bundle_summary,
        _vendor_score,
        _amount_score,
        _item_overlap_score,
    )
    from alibi.db.models import BundleType

    # Load bundle data for the moved bundle
    cloud_id = target_cloud_id or source_cloud_id
    if not cloud_id:
        return
    bundle_data = v2_store.get_cloud_bundle_data(db, cloud_id)
    if not bundle_data:
        return

    # Find the moved bundle and another bundle for comparison
    moved_atoms: list[dict[str, Any]] = []
    other_atoms: list[dict[str, Any]] = []
    moved_type = "basket"
    other_type = None

    for b in bundle_data:
        if b.get("bundle_id") == bundle_id:
            moved_atoms = b.get("atoms", [])
            moved_type = b.get("bundle_type", "basket")
        else:
            if not other_atoms:
                other_atoms = b.get("atoms", [])
                other_type = b.get("bundle_type")

    if not moved_atoms:
        return

    # Build summaries for scoring
    moved_summary = extract_bundle_summary(
        bundle_id,
        BundleType(moved_type) if moved_type else BundleType.BASKET,
        moved_atoms,
    )

    # Extract source cloud vendor info from other bundles
    vendor_key_a = moved_summary.vendor_key
    vendor_key_b = None
    vendor_similarity = 0.0
    amount_diff = 0.0
    date_diff_days = 0
    location_distance = None
    item_overlap = 0.0

    if other_atoms:
        other_summary = extract_bundle_summary(
            "other",
            BundleType(other_type) if other_type else BundleType.BASKET,
            other_atoms,
        )
        vendor_key_b = other_summary.vendor_key

        # Compute feature values
        vs = _vendor_score(moved_summary, other_summary)
        vendor_similarity = float(vs)

        _amount_result, _ = _amount_score(moved_summary, other_summary)
        if moved_summary.amount and other_summary.amount:
            amount_diff = float(abs(moved_summary.amount - other_summary.amount))

        if moved_summary.event_date and other_summary.event_date:
            date_diff_days = abs(
                (moved_summary.event_date - other_summary.event_date).days
            )

        # Location distance
        if (
            moved_summary.lat
            and moved_summary.lng
            and other_summary.lat
            and other_summary.lng
        ):
            from alibi.utils.map_url import haversine_distance

            location_distance = haversine_distance(
                moved_summary.lat,
                moved_summary.lng,
                other_summary.lat,
                other_summary.lng,
            )

        io = _item_overlap_score(moved_summary, other_summary)
        item_overlap = float(io)

    feature = CorrectionFeatureVector(
        vendor_key_a=vendor_key_a,
        vendor_key_b=vendor_key_b,
        vendor_similarity=vendor_similarity,
        amount_diff=amount_diff,
        date_diff_days=date_diff_days,
        location_distance=location_distance,
        was_false_positive=was_false_positive,
        source_bundle_type=moved_type,
        target_bundle_type=other_type,
        item_overlap=item_overlap,
    )
    record_correction(db, feature)


def _teach_identity_from_correction(
    db: DatabaseManager,
    moved_bundle_id: str,
    target_cloud_id: str | None,
) -> None:
    """When a bundle is moved into a cloud, teach identity system about the merge.

    If the moved bundle's vendor differs from the target cloud's vendor,
    link them under the same identity (source="correction").
    """
    if not target_cloud_id:
        return

    from alibi.db.models import AtomType
    from alibi.identities.matching import ensure_vendor_identity

    def _extract_vendor_from_bundles(
        bundles: list[dict[str, Any]], exclude_bundle_id: str | None = None
    ) -> tuple[str | None, str | None]:
        """Extract vendor name and VAT from bundle atom data."""
        for b in bundles:
            if exclude_bundle_id and b.get("bundle_id") == exclude_bundle_id:
                continue
            for atom in b.get("atoms", []):
                atype = atom.get("atom_type", "")
                if atype == AtomType.VENDOR.value or atype == "vendor":
                    data = atom.get("data", {})
                    return data.get("name"), data.get("vat_number")
        return None, None

    # Load all bundles in target cloud (includes the moved bundle now)
    cloud_data = v2_store.get_cloud_bundle_data(db, target_cloud_id)
    if not cloud_data:
        return

    # Extract vendor from moved bundle
    moved_vendor, moved_vat = _extract_vendor_from_bundles(
        [b for b in cloud_data if b.get("bundle_id") == moved_bundle_id]
    )
    # Extract vendor from other bundles in the cloud
    target_vendor, target_vat = _extract_vendor_from_bundles(
        cloud_data, exclude_bundle_id=moved_bundle_id
    )

    if not moved_vendor and not target_vendor:
        return

    # Ensure both vendors are in the same identity.
    # First call creates/finds, second call adds as member of same or new.
    identity_id = ensure_vendor_identity(
        db,
        vendor_name=target_vendor,
        vendor_key=target_vat,
        vat_number=target_vat,
        source="correction",
    )
    if identity_id and moved_vendor:
        ensure_vendor_identity(
            db,
            vendor_name=moved_vendor,
            vendor_key=moved_vat,
            vat_number=moved_vat,
            source="correction",
        )

    logger.info(
        f"Identity feedback: linked '{moved_vendor}' with '{target_vendor}' "
        f"(correction)"
    )


def _try_recollapse(db: DatabaseManager, cloud_id: str) -> str | None:
    """Run collapse logic on a cloud and store the result if it collapses.

    Returns the fact ID if collapsed, None otherwise.
    """
    bundles = v2_store.get_cloud_bundle_data(db, cloud_id)
    if not bundles:
        return None

    cloud = Cloud(id=cloud_id, status=CloudStatus.FORMING)
    collapse_result = try_collapse(cloud, bundles)

    if collapse_result.collapsed and collapse_result.fact:
        fact = collapse_result.fact
        # Set fact_id on all items
        for item in collapse_result.items:
            item.fact_id = fact.id
        v2_store.store_fact(db, fact, collapse_result.items)
        return fact.id
    else:
        v2_store.set_cloud_status(db, cloud_id, collapse_result.cloud_status.value)
        return None
