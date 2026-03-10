"""Tests for alibi.services.correction — correction service facade."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alibi.clouds.correction import CorrectionResult
from alibi.services import correction as svc


# ---------------------------------------------------------------------------
# move_bundle — delegates to clouds.correction.move_bundle
# ---------------------------------------------------------------------------


class TestMoveBundleFacade:
    def test_delegates_to_clouds_correction(self, mock_db):
        expected = CorrectionResult(success=True, target_cloud_id="cloud-1")

        with patch(
            "alibi.services.correction._correction.move_bundle",
            return_value=expected,
        ) as mock_fn:
            result = svc.move_bundle(mock_db, "bundle-abc")

        mock_fn.assert_called_once_with(mock_db, "bundle-abc", None)
        assert result is expected

    def test_passes_target_cloud_id(self, mock_db):
        expected = CorrectionResult(success=True)

        with patch(
            "alibi.services.correction._correction.move_bundle",
            return_value=expected,
        ) as mock_fn:
            result = svc.move_bundle(mock_db, "bundle-xyz", target_cloud_id="cloud-99")

        mock_fn.assert_called_once_with(mock_db, "bundle-xyz", "cloud-99")
        assert result is expected

    def test_returns_failure_result_unchanged(self, mock_db):
        failure = CorrectionResult(
            success=False, error="Bundle xyz is not assigned to any cloud"
        )

        with patch(
            "alibi.services.correction._correction.move_bundle",
            return_value=failure,
        ):
            result = svc.move_bundle(mock_db, "bundle-xyz")

        assert result.success is False
        assert "not assigned" in (result.error or "")


# ---------------------------------------------------------------------------
# recollapse_cloud — wraps clouds.correction.recollapse_cloud
# ---------------------------------------------------------------------------


class TestRecollapseCloud:
    def test_returns_correction_result_on_success(self, mock_db):
        with patch(
            "alibi.services.correction._correction.recollapse_cloud",
            return_value="fact-new-1",
        ):
            result = svc.recollapse_cloud(mock_db, "cloud-42")

        assert isinstance(result, CorrectionResult)
        assert result.success is True
        assert result.target_cloud_id == "cloud-42"
        assert result.target_fact_id == "fact-new-1"

    def test_returns_none_fact_id_when_cloud_stays_forming(self, mock_db):
        with patch(
            "alibi.services.correction._correction.recollapse_cloud",
            return_value=None,
        ):
            result = svc.recollapse_cloud(mock_db, "cloud-forming")

        assert result.success is True
        assert result.target_fact_id is None

    def test_delegates_with_correct_args(self, mock_db):
        with patch(
            "alibi.services.correction._correction.recollapse_cloud",
            return_value=None,
        ) as mock_fn:
            svc.recollapse_cloud(mock_db, "cloud-77")

        mock_fn.assert_called_once_with(mock_db, "cloud-77")


# ---------------------------------------------------------------------------
# mark_disputed — wraps clouds.correction.mark_disputed
# ---------------------------------------------------------------------------


class TestMarkDisputed:
    def test_returns_correction_result_on_success(self, mock_db):
        with patch(
            "alibi.services.correction._correction.mark_disputed",
            return_value=True,
        ):
            result = svc.mark_disputed(mock_db, "cloud-disputed")

        assert isinstance(result, CorrectionResult)
        assert result.success is True
        assert result.source_cloud_id == "cloud-disputed"

    def test_delegates_with_correct_args(self, mock_db):
        with patch(
            "alibi.services.correction._correction.mark_disputed",
            return_value=True,
        ) as mock_fn:
            svc.mark_disputed(mock_db, "cloud-99")

        mock_fn.assert_called_once_with(mock_db, "cloud-99")


# ---------------------------------------------------------------------------
# update_fact — field allowlist enforcement + DB interaction
# ---------------------------------------------------------------------------


class TestUpdateFact:
    def test_returns_false_when_fact_not_found(self, mock_db):
        with patch(
            "alibi.services.correction.v2_store.get_fact_by_id",
            return_value=None,
        ):
            result = svc.update_fact(mock_db, "fact-missing", {"vendor": "Acme"})

        assert result is False

    def test_rejects_disallowed_field(self, mock_db):
        with pytest.raises(ValueError, match="Disallowed field"):
            svc.update_fact(mock_db, "fact-1", {"secret_field": "x"})

    def test_rejects_empty_fields(self, mock_db):
        with pytest.raises(ValueError, match="must not be empty"):
            svc.update_fact(mock_db, "fact-1", {})

    def test_rejects_mix_of_allowed_and_disallowed(self, mock_db):
        with pytest.raises(ValueError, match="Disallowed field"):
            svc.update_fact(mock_db, "fact-1", {"vendor": "Acme", "bad_field": "x"})

    def test_updates_single_allowed_field(self, db_manager):
        """Integration test against real DB — insert a fact then update it."""
        from alibi.db import v2_store
        from alibi.db.models import (
            Cloud,
            CloudStatus,
            Fact,
            FactStatus,
            FactType,
        )
        import uuid
        from datetime import date

        # Insert prerequisite cloud
        cloud_id = str(uuid.uuid4())
        with db_manager.transaction() as cursor:
            cursor.execute(
                "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
                (cloud_id, CloudStatus.COLLAPSED.value, 1.0),
            )

        # Insert a fact
        fact_id = str(uuid.uuid4())
        fact = Fact(
            id=fact_id,
            cloud_id=cloud_id,
            fact_type=FactType.PURCHASE,
            vendor="Old Vendor",
            vendor_key=None,
            total_amount=Decimal("42.00"),
            currency="EUR",
            event_date=date(2024, 1, 15),
            status=FactStatus.CONFIRMED,
        )
        v2_store.store_fact(db_manager, fact, [])

        result = svc.update_fact(db_manager, fact_id, {"vendor": "New Vendor"})

        assert result is True
        updated = v2_store.get_fact_by_id(db_manager, fact_id)
        assert updated is not None
        assert updated["vendor"] == "New Vendor"

    def test_updates_amount_field(self, db_manager):
        """Verify that 'amount' maps to the total_amount column."""
        from alibi.db import v2_store
        from alibi.db.models import (
            Cloud,
            CloudStatus,
            Fact,
            FactStatus,
            FactType,
        )
        import uuid
        from datetime import date

        cloud_id = str(uuid.uuid4())
        with db_manager.transaction() as cursor:
            cursor.execute(
                "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
                (cloud_id, CloudStatus.COLLAPSED.value, 1.0),
            )

        fact_id = str(uuid.uuid4())
        fact = Fact(
            id=fact_id,
            cloud_id=cloud_id,
            fact_type=FactType.PURCHASE,
            vendor="Shop",
            vendor_key=None,
            total_amount=Decimal("10.00"),
            currency="EUR",
            event_date=date(2024, 3, 1),
            status=FactStatus.CONFIRMED,
        )
        v2_store.store_fact(db_manager, fact, [])

        result = svc.update_fact(db_manager, fact_id, {"amount": 99.99})

        assert result is True
        updated = v2_store.get_fact_by_id(db_manager, fact_id)
        assert updated is not None
        assert float(updated["total_amount"]) == pytest.approx(99.99)

    def test_all_allowed_fields_accepted(self, mock_db):
        """Verify the full allowlist is accepted without ValueError."""
        existing_fact = {"id": "fact-1", "vendor": "Old", "vendor_key": None}
        with patch(
            "alibi.services.correction.v2_store.get_fact_by_id",
            return_value=existing_fact,
        ):
            with patch.object(mock_db, "transaction") as mock_tx:
                mock_cursor = MagicMock()
                mock_tx.return_value.__enter__ = MagicMock(return_value=mock_cursor)
                mock_tx.return_value.__exit__ = MagicMock(return_value=False)

                # Each allowed field individually — just check no ValueError
                for field_name in (
                    "vendor",
                    "amount",
                    "date",
                    "fact_type",
                    "vendor_key",
                ):
                    result = svc.update_fact(mock_db, "fact-1", {field_name: "x"})
                    assert result is True


# ---------------------------------------------------------------------------
# correct_vendor — higher-level: update fact + teach identity system
# ---------------------------------------------------------------------------


class TestCorrectVendor:
    def test_returns_false_when_fact_not_found(self, mock_db):
        with patch(
            "alibi.services.correction.v2_store.get_fact_by_id",
            return_value=None,
        ):
            result = svc.correct_vendor(mock_db, "fact-missing", "New Name")

        assert result is False

    def test_updates_vendor_and_calls_identity(self, mock_db):
        existing_fact = {
            "id": "fact-1",
            "vendor": "Old Corp",
            "vendor_key": "CY10057000Y",
        }

        with (
            patch(
                "alibi.services.correction.v2_store.get_fact_by_id",
                return_value=existing_fact,
            ),
            patch.object(mock_db, "transaction") as mock_tx,
            patch("alibi.services.correction.ensure_vendor_identity") as mock_identity,
        ):
            mock_cursor = MagicMock()
            mock_tx.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_tx.return_value.__exit__ = MagicMock(return_value=False)

            result = svc.correct_vendor(mock_db, "fact-1", "NEW CORP LTD")

        assert result is True
        # Identity system was called with normalized name + existing vendor_key
        mock_identity.assert_called_once()
        call_kwargs = mock_identity.call_args.kwargs
        assert call_kwargs["vendor_key"] == "CY10057000Y"
        assert call_kwargs["source"] == "correction"
        # Normalized form strips LTD suffix and title-cases
        assert (
            "Corp" in call_kwargs["vendor_name"] or "New" in call_kwargs["vendor_name"]
        )

    def test_normalizes_vendor_name_before_storing(self, mock_db):
        """Vendor name written to DB is the normalized (title-case) form."""
        existing_fact = {"id": "fact-2", "vendor": "Old Shop", "vendor_key": None}
        captured_sql = []
        captured_params = []

        def fake_transaction():
            ctx = MagicMock()
            mock_cursor = MagicMock()

            def execute(sql, params=()):
                captured_sql.append(sql)
                captured_params.append(params)

            mock_cursor.execute = execute
            ctx.__enter__ = MagicMock(return_value=mock_cursor)
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        with (
            patch(
                "alibi.services.correction.v2_store.get_fact_by_id",
                return_value=existing_fact,
            ),
            patch(
                "alibi.services.correction.ensure_vendor_identity",
                return_value="identity-1",
            ),
        ):
            mock_db.transaction = fake_transaction
            result = svc.correct_vendor(mock_db, "fact-2", "BEST FOODS LLC")

        assert result is True
        assert len(captured_params) >= 1
        stored_vendor = captured_params[0][0]
        # Normalized: strips LLC suffix, title-cases
        assert stored_vendor == "Best Foods"

    def test_correct_vendor_integration(self, db_manager):
        """Full integration: updates vendor on real DB row and registers identity."""
        from alibi.db import v2_store
        from alibi.db.models import (
            Cloud,
            CloudStatus,
            Fact,
            FactStatus,
            FactType,
        )
        import uuid
        from datetime import date

        cloud_id = str(uuid.uuid4())
        with db_manager.transaction() as cursor:
            cursor.execute(
                "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
                (cloud_id, CloudStatus.COLLAPSED.value, 1.0),
            )

        fact_id = str(uuid.uuid4())
        fact = Fact(
            id=fact_id,
            cloud_id=cloud_id,
            fact_type=FactType.PURCHASE,
            vendor="Original Vendor",
            vendor_key=None,
            total_amount=Decimal("15.00"),
            currency="EUR",
            event_date=date(2024, 6, 10),
            status=FactStatus.CONFIRMED,
        )
        v2_store.store_fact(db_manager, fact, [])

        result = svc.correct_vendor(db_manager, fact_id, "Updated Vendor Ltd")

        assert result is True
        updated = v2_store.get_fact_by_id(db_manager, fact_id)
        assert updated is not None
        # Normalized: strips Ltd, title-cases
        assert updated["vendor"] == "Updated Vendor"

    def test_correct_vendor_updates_identity_canonical_name(self, db_manager):
        """correct_vendor() propagates the new name to the identity canonical_name.

        Uses vendor_key to link the fact to the existing identity, matching the
        real-world scenario where a vendor has a known VAT number.
        """
        from alibi.db import v2_store
        from alibi.db.models import (
            CloudStatus,
            Fact,
            FactStatus,
            FactType,
        )
        from alibi.identities import store as id_store
        from alibi.normalizers.vendors import normalize_vendor
        import uuid
        from datetime import date

        vendor_key = "CY10180201N"
        new_vendor_input = "Mediterranean Hospital of Cyprus"

        # Seed an identity with a stale canonical name, linked via vendor_key
        identity_id = id_store.create_identity(
            db_manager, "vendor", "Mediterranean Hospital C"
        )
        id_store.add_member(db_manager, identity_id, "vendor_key", vendor_key)

        # Create a fact with the old (stale) vendor name and the same vendor_key
        cloud_id = str(uuid.uuid4())
        with db_manager.transaction() as cursor:
            cursor.execute(
                "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
                (cloud_id, CloudStatus.COLLAPSED.value, 1.0),
            )
        fact_id = str(uuid.uuid4())
        fact = Fact(
            id=fact_id,
            cloud_id=cloud_id,
            fact_type=FactType.PURCHASE,
            vendor="Mediterranean Hospital C",
            vendor_key=vendor_key,
            total_amount=Decimal("120.00"),
            currency="EUR",
            event_date=date(2024, 9, 1),
            status=FactStatus.CONFIRMED,
        )
        v2_store.store_fact(db_manager, fact, [])

        # Correct the vendor name
        result = svc.correct_vendor(db_manager, fact_id, new_vendor_input)

        assert result is True

        # Verify the fact vendor was updated to the normalized form
        updated = v2_store.get_fact_by_id(db_manager, fact_id)
        assert updated is not None
        assert updated["vendor"] == normalize_vendor(new_vendor_input)

        # Verify the identity canonical_name was updated (found via vendor_key)
        identity = id_store.get_identity(db_manager, identity_id)
        assert identity is not None
        assert identity["canonical_name"] == normalize_vendor(new_vendor_input)
        assert identity["canonical_name"] != "Mediterranean Hospital C"

    def test_correct_vendor_stale_canonical_name_scenario(self, db_manager):
        """Repeated correction keeps refining canonical_name to latest value.

        Uses vendor_key to ensure consistent identity resolution across corrections.
        """
        from alibi.db import v2_store
        from alibi.db.models import (
            CloudStatus,
            Fact,
            FactStatus,
            FactType,
        )
        from alibi.identities import store as id_store
        from alibi.normalizers.vendors import normalize_vendor
        import uuid
        from datetime import date

        vendor_key = "CY10180201N"
        first_correction = "Mediterranean Hospital C"
        second_correction = "Mediterranean Hospital of Cyprus"

        # Start with an identity whose canonical_name is very stale
        identity_id = id_store.create_identity(db_manager, "vendor", "Med Hosp")
        id_store.add_member(db_manager, identity_id, "vendor_key", vendor_key)

        cloud_id = str(uuid.uuid4())
        with db_manager.transaction() as cursor:
            cursor.execute(
                "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
                (cloud_id, CloudStatus.COLLAPSED.value, 1.0),
            )
        fact_id = str(uuid.uuid4())
        fact = Fact(
            id=fact_id,
            cloud_id=cloud_id,
            fact_type=FactType.PURCHASE,
            vendor="Med Hosp",
            vendor_key=vendor_key,
            total_amount=Decimal("50.00"),
            currency="EUR",
            event_date=date(2024, 10, 1),
            status=FactStatus.CONFIRMED,
        )
        v2_store.store_fact(db_manager, fact, [])

        # First correction: partial improvement
        svc.correct_vendor(db_manager, fact_id, first_correction)
        identity = id_store.get_identity(db_manager, identity_id)
        assert identity["canonical_name"] == normalize_vendor(first_correction)

        # Second correction: full canonical name
        svc.correct_vendor(db_manager, fact_id, second_correction)
        identity = id_store.get_identity(db_manager, identity_id)
        assert identity["canonical_name"] == normalize_vendor(second_correction)
