"""Unit tests for the shared enrichment-batch helpers (_batch.py)."""

from __future__ import annotations

from unittest.mock import patch

from alibi.enrichment._batch import call_enrichment_llm, run_vendor_batches


class TestCallEnrichmentLlm:
    def test_returns_items_list(self) -> None:
        with patch(
            "alibi.extraction.structurer.structure_ocr_text",
            return_value={"items": [{"idx": 1}, {"idx": 2}]},
        ):
            out = call_enrichment_llm("p", timeout=1.0)
        assert out == [{"idx": 1}, {"idx": 2}]

    def test_llm_exception_returns_empty(self) -> None:
        with patch(
            "alibi.extraction.structurer.structure_ocr_text",
            side_effect=RuntimeError("boom"),
        ):
            assert call_enrichment_llm("p", timeout=1.0) == []

    def test_non_list_items_returns_empty(self) -> None:
        with patch(
            "alibi.extraction.structurer.structure_ocr_text",
            return_value={"items": {"not": "a list"}},
        ):
            assert call_enrichment_llm("p", timeout=1.0) == []

    def test_response_format_threaded_to_structurer(self) -> None:
        schema = {"type": "object"}
        with patch(
            "alibi.extraction.structurer.structure_ocr_text",
            return_value={"items": []},
        ) as mock:
            call_enrichment_llm("p", timeout=1.0, response_format=schema)
        assert mock.call_args.kwargs["response_format"] is schema

    def test_response_format_defaults_none(self) -> None:
        with patch(
            "alibi.extraction.structurer.structure_ocr_text",
            return_value={"items": []},
        ) as mock:
            call_enrichment_llm("p", timeout=1.0)
        assert mock.call_args.kwargs["response_format"] is None


class TestReformatRetry:
    _ASK = "No JSON found in response: Please provide the list of items you would like"

    def test_retries_with_reinforced_prompt_and_succeeds(self) -> None:
        with patch(
            "alibi.extraction.structurer.structure_ocr_text",
            side_effect=[
                RuntimeError(self._ASK),
                {"items": [{"idx": 1}]},
            ],
        ) as mock_llm:
            out = call_enrichment_llm("PROMPT", timeout=1.0)
        assert out == [{"idx": 1}]
        assert mock_llm.call_count == 2
        # The retry re-asserts the items are already present.
        retry_prompt = mock_llm.call_args_list[1].kwargs["emphasis_prompt"]
        assert retry_prompt.startswith("PROMPT")
        assert "already listed above" in retry_prompt
        assert "ONLY the JSON" in retry_prompt

    def test_generic_failure_is_not_retried(self) -> None:
        with patch(
            "alibi.extraction.structurer.structure_ocr_text",
            side_effect=RuntimeError("connection refused"),
        ) as mock_llm:
            assert call_enrichment_llm("p", timeout=1.0) == []
        assert mock_llm.call_count == 1

    def test_unrelated_prose_is_not_retried(self) -> None:
        # A no-JSON failure that is NOT the "give me the items" variant.
        with patch(
            "alibi.extraction.structurer.structure_ocr_text",
            side_effect=RuntimeError("No JSON found in response: I cannot help."),
        ) as mock_llm:
            assert call_enrichment_llm("p", timeout=1.0) == []
        assert mock_llm.call_count == 1

    def test_retry_failing_again_returns_empty_without_looping(self) -> None:
        with patch(
            "alibi.extraction.structurer.structure_ocr_text",
            side_effect=[RuntimeError(self._ASK), RuntimeError(self._ASK)],
        ) as mock_llm:
            assert call_enrichment_llm("p", timeout=1.0) == []
        assert mock_llm.call_count == 2  # one retry only, no infinite loop


class TestRunVendorBatches:
    def test_groups_by_vendor_and_slices_into_batches(self) -> None:
        rows = [
            {"id": "a", "vendor": "V1"},
            {"id": "b", "vendor": "V1"},
            {"id": "c", "vendor": "V1"},
            {"id": "d", "vendor": "V2"},
        ]
        calls: list[tuple[str, int]] = []

        def enrich_batch(vendor: str, items: list[dict]) -> list[str]:
            calls.append((vendor, len(items)))
            return [i["id"] for i in items]

        out = run_vendor_batches(rows, batch_size=2, enrich_batch=enrich_batch)

        # V1 (3 items) -> two sub-batches of 2 + 1; V2 (1 item) -> one batch.
        assert calls == [("V1", 2), ("V1", 1), ("V2", 1)]
        assert out == ["a", "b", "c", "d"]

    def test_missing_vendor_falls_back_to_unknown(self) -> None:
        seen: list[str] = []
        run_vendor_batches(
            [{"id": "a", "vendor": None}, {"id": "b"}],
            batch_size=10,
            enrich_batch=lambda vendor, items: seen.append(vendor) or [],
        )
        assert seen == ["Unknown"]

    def test_empty_rows_returns_empty(self) -> None:
        assert run_vendor_batches([], 10, lambda v, i: [1]) == []
