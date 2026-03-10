"""Location analytics: spending by location, vendor branches, suggestions.

Analyzes location-annotated facts to provide geographic spending insights.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LocationSpending:
    """Spending at a specific location."""

    lat: float
    lng: float
    place_name: str | None
    map_url: str
    total_amount: float
    visit_count: int
    vendors: list[str]
    avg_amount: float = 0.0
    first_visit: str | None = None
    last_visit: str | None = None

    def __post_init__(self) -> None:
        if self.visit_count > 0:
            self.avg_amount = round(self.total_amount / self.visit_count, 2)


@dataclass
class VendorBranch:
    """A vendor branch at a specific location."""

    vendor_key: str
    vendor_name: str
    lat: float
    lng: float
    place_name: str | None
    map_url: str
    visit_count: int
    total_spent: float
    avg_basket: float = 0.0
    last_visit: str | None = None


@dataclass
class VendorBranchComparison:
    """Comparison of a vendor's branches."""

    vendor_key: str
    vendor_name: str
    branch_count: int
    branches: list[VendorBranch]
    total_spent: float
    most_visited: VendorBranch | None = None
    highest_avg: VendorBranch | None = None


@dataclass
class LocationSuggestion:
    """Location-aware vendor suggestion."""

    vendor_name: str
    vendor_key: str
    lat: float
    lng: float
    place_name: str | None
    distance_meters: float
    visit_count: int
    avg_basket: float
    reason: str


def _load_location_facts(db: Any) -> list[dict[str, Any]]:
    """Load all facts with location annotations."""
    rows = db.fetchall(
        """
        SELECT f.id, f.event_type, f.total_amount, f.event_date, f.vendor_key,
               a.metadata, a.value as map_url
        FROM facts f
        JOIN annotations a ON a.target_type = 'fact' AND a.target_id = f.id
        WHERE a.annotation_type = 'location'
        AND f.total_amount IS NOT NULL
        """,
        (),
    )
    results = []
    for row in rows:
        metadata = row[5]
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                continue
        if not metadata or "lat" not in metadata or "lng" not in metadata:
            continue

        results.append(
            {
                "fact_id": row[0],
                "event_type": row[1],
                "total_amount": float(row[2]) if row[2] else 0.0,
                "event_date": str(row[3]) if row[3] else None,
                "vendor_key": row[4],
                "lat": float(metadata["lat"]),
                "lng": float(metadata["lng"]),
                "place_name": metadata.get("place_name"),
                "map_url": row[6],
            }
        )
    return results


def _get_vendor_name(db: Any, vendor_key: str) -> str:
    """Resolve vendor name from identity system."""
    name_row = db.fetchone(
        """
        SELECT im.value FROM identity_members im
        JOIN identities i ON im.identity_id = i.id
        WHERE i.entity_type = 'vendor'
        AND im.member_type = 'name'
        AND im.identity_id = (
            SELECT identity_id FROM identity_members
            WHERE member_type = 'vendor_key' AND value = ?
            LIMIT 1
        )
        LIMIT 1
        """,
        (vendor_key,),
    )
    return name_row[0] if name_row else vendor_key


def spending_by_location(
    db: Any,
    cluster_radius_m: float = 100.0,
) -> list[LocationSpending]:
    """Aggregate spending by location (clustered by radius).

    Args:
        db: DatabaseManager.
        cluster_radius_m: Cluster locations within this radius (meters).

    Returns:
        List of LocationSpending sorted by total_amount descending.
    """
    from alibi.utils.map_url import haversine_distance

    facts = _load_location_facts(db)
    if not facts:
        return []

    # Simple greedy clustering
    clusters: list[dict[str, Any]] = []
    assigned: set[int] = set()

    for i, f in enumerate(facts):
        if i in assigned:
            continue

        cluster: dict[str, Any] = {
            "lat": f["lat"],
            "lng": f["lng"],
            "place_name": f["place_name"],
            "map_url": f["map_url"],
            "facts": [f],
        }
        assigned.add(i)

        for j in range(i + 1, len(facts)):
            if j in assigned:
                continue
            dist = haversine_distance(
                f["lat"], f["lng"], facts[j]["lat"], facts[j]["lng"]
            )
            if dist <= cluster_radius_m:
                cluster["facts"].append(facts[j])
                assigned.add(j)

        clusters.append(cluster)

    results = []
    for c in clusters:
        vendors = list({f["vendor_key"] or "unknown" for f in c["facts"]})
        dates = [f["event_date"] for f in c["facts"] if f["event_date"]]
        total = sum(f["total_amount"] for f in c["facts"])

        results.append(
            LocationSpending(
                lat=c["lat"],
                lng=c["lng"],
                place_name=c["place_name"],
                map_url=c["map_url"],
                total_amount=round(total, 2),
                visit_count=len(c["facts"]),
                vendors=vendors,
                first_visit=min(dates) if dates else None,
                last_visit=max(dates) if dates else None,
            )
        )

    return sorted(results, key=lambda x: x.total_amount, reverse=True)


def vendor_branch_comparison(
    db: Any,
    vendor_key: str | None = None,
) -> list[VendorBranchComparison]:
    """Compare spending across vendor branches (different locations).

    Args:
        db: DatabaseManager.
        vendor_key: Filter to specific vendor, or None for all.

    Returns:
        List of VendorBranchComparison for vendors with 2+ branches.
    """
    from alibi.utils.map_url import haversine_distance

    facts = _load_location_facts(db)
    if not facts:
        return []

    # Group by vendor_key + location
    vendor_locations: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for f in facts:
        vk = f["vendor_key"]
        if not vk:
            continue
        if vendor_key and vk != vendor_key:
            continue
        loc_key = f"{f['lat']:.5f},{f['lng']:.5f}"
        vendor_locations[vk][loc_key].append(f)

    results = []
    for vk, locations in vendor_locations.items():
        if len(locations) < 2 and not vendor_key:
            continue

        # Cluster nearby locations (within 200m)
        loc_items = list(locations.items())
        merged: list[list[dict[str, Any]]] = []
        used: set[int] = set()

        for i, (_key_i, facts_i) in enumerate(loc_items):
            if i in used:
                continue
            group = list(facts_i)
            used.add(i)
            lat_i = facts_i[0]["lat"]
            lng_i = facts_i[0]["lng"]

            for j, (_key_j, facts_j) in enumerate(loc_items):
                if j in used:
                    continue
                lat_j = facts_j[0]["lat"]
                lng_j = facts_j[0]["lng"]
                if haversine_distance(lat_i, lng_i, lat_j, lng_j) < 200:
                    group.extend(facts_j)
                    used.add(j)

            merged.append(group)

        if len(merged) < 2 and not vendor_key:
            continue

        vendor_name = _get_vendor_name(db, vk)

        branches = []
        for group in merged:
            total = sum(f["total_amount"] for f in group)
            dates = [f["event_date"] for f in group if f["event_date"]]
            branches.append(
                VendorBranch(
                    vendor_key=vk,
                    vendor_name=vendor_name,
                    lat=group[0]["lat"],
                    lng=group[0]["lng"],
                    place_name=group[0].get("place_name"),
                    map_url=group[0]["map_url"],
                    visit_count=len(group),
                    total_spent=round(total, 2),
                    avg_basket=(round(total / len(group), 2) if group else 0),
                    last_visit=max(dates) if dates else None,
                )
            )

        branches.sort(key=lambda b: b.total_spent, reverse=True)
        total_all = sum(b.total_spent for b in branches)

        comparison = VendorBranchComparison(
            vendor_key=vk,
            vendor_name=vendor_name,
            branch_count=len(branches),
            branches=branches,
            total_spent=round(total_all, 2),
            most_visited=(
                max(branches, key=lambda b: b.visit_count) if branches else None
            ),
            highest_avg=(
                max(branches, key=lambda b: b.avg_basket) if branches else None
            ),
        )
        results.append(comparison)

    return sorted(results, key=lambda x: x.total_spent, reverse=True)


def nearby_vendor_suggestions(
    db: Any,
    lat: float,
    lng: float,
    radius_m: float = 2000.0,
    limit: int = 10,
) -> list[LocationSuggestion]:
    """Suggest vendors near a location based on visit history.

    Args:
        db: DatabaseManager.
        lat: Current latitude.
        lng: Current longitude.
        radius_m: Search radius in meters.
        limit: Max suggestions.

    Returns:
        List of LocationSuggestion sorted by distance.
    """
    from alibi.utils.map_url import haversine_distance

    facts = _load_location_facts(db)
    if not facts:
        return []

    # Group by vendor
    vendor_locs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in facts:
        vk = f["vendor_key"]
        if vk:
            vendor_locs[vk].append(f)

    suggestions = []
    seen_vendors: set[str] = set()

    for vk, vf in vendor_locs.items():
        # Find closest branch
        closest = None
        min_dist = float("inf")

        for f in vf:
            dist = haversine_distance(lat, lng, f["lat"], f["lng"])
            if dist < min_dist:
                min_dist = dist
                closest = f

        if not closest or min_dist > radius_m:
            continue

        if vk in seen_vendors:
            continue
        seen_vendors.add(vk)

        total = sum(f["total_amount"] for f in vf)
        avg = total / len(vf)

        vendor_name = _get_vendor_name(db, vk)

        reason = f"{len(vf)} previous visits, avg basket {avg:.2f}"
        if min_dist < 500:
            reason = f"Very close ({min_dist:.0f}m). " + reason

        suggestions.append(
            LocationSuggestion(
                vendor_name=vendor_name,
                vendor_key=vk,
                lat=closest["lat"],
                lng=closest["lng"],
                place_name=closest.get("place_name"),
                distance_meters=round(min_dist, 1),
                visit_count=len(vf),
                avg_basket=round(avg, 2),
                reason=reason,
            )
        )

    suggestions.sort(key=lambda s: s.distance_meters)
    return suggestions[:limit]
