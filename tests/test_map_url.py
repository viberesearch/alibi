"""Tests for Google Maps URL parser and haversine distance."""

import math

import pytest

from alibi.utils.map_url import (
    haversine_distance,
    is_map_url,
    parse_map_url,
)


class TestParseMapUrl:
    """Test parse_map_url with various Google Maps URL formats."""

    def test_place_url_with_at_coords(self):
        url = "https://www.google.com/maps/place/Alphamega+Paphos/@34.7724,32.4218,17z"
        result = parse_map_url(url)
        assert result is not None
        assert result["lat"] == pytest.approx(34.7724)
        assert result["lng"] == pytest.approx(32.4218)
        assert result["place_name"] == "Alphamega Paphos"

    def test_place_url_with_data_suffix(self):
        url = (
            "https://www.google.com/maps/place/Lidl/@34.6789,33.0456,17z"
            "/data=!3m1!4b1!4m6"
        )
        result = parse_map_url(url)
        assert result is not None
        assert result["lat"] == pytest.approx(34.6789)
        assert result["lng"] == pytest.approx(33.0456)
        assert result["place_name"] == "Lidl"

    def test_direct_at_url(self):
        url = "https://www.google.com/maps/@34.7724,32.4218,15z"
        result = parse_map_url(url)
        assert result is not None
        assert result["lat"] == pytest.approx(34.7724)
        assert result["lng"] == pytest.approx(32.4218)
        assert result["place_name"] is None

    def test_q_parameter_coords(self):
        url = "https://maps.google.com/maps?q=34.7724,32.4218"
        result = parse_map_url(url)
        assert result is not None
        assert result["lat"] == pytest.approx(34.7724)
        assert result["lng"] == pytest.approx(32.4218)

    def test_q_parameter_with_space(self):
        url = "https://maps.google.com/maps?q=34.7724, 32.4218"
        result = parse_map_url(url)
        assert result is not None
        assert result["lat"] == pytest.approx(34.7724)
        assert result["lng"] == pytest.approx(32.4218)

    def test_ll_parameter(self):
        url = "https://maps.google.com/maps?ll=34.7724,32.4218&z=15"
        result = parse_map_url(url)
        assert result is not None
        assert result["lat"] == pytest.approx(34.7724)
        assert result["lng"] == pytest.approx(32.4218)

    def test_negative_coords(self):
        url = "https://www.google.com/maps/@-33.8688,151.2093,15z"
        result = parse_map_url(url)
        assert result is not None
        assert result["lat"] == pytest.approx(-33.8688)
        assert result["lng"] == pytest.approx(151.2093)

    def test_country_specific_domain(self):
        url = "https://maps.google.gr/maps?q=34.7724,32.4218"
        result = parse_map_url(url)
        assert result is not None
        assert result["lat"] == pytest.approx(34.7724)

    def test_strips_tracking_params(self):
        url = (
            "https://www.google.com/maps/place/Shop/@34.77,32.42,17z"
            "?utm_source=share&utm_medium=link&entry=gps"
        )
        result = parse_map_url(url)
        assert result is not None
        assert "utm_source" not in result["clean_url"]
        assert "utm_medium" not in result["clean_url"]
        assert "entry" not in result["clean_url"]

    def test_preserves_non_tracking_params(self):
        url = "https://maps.google.com/maps?q=34.77,32.42&z=15"
        result = parse_map_url(url)
        assert result is not None
        assert "z=15" in result["clean_url"]

    def test_url_encoded_place_name(self):
        url = "https://www.google.com/maps/place/My%20Shop/@34.77,32.42,17z"
        result = parse_map_url(url)
        assert result is not None
        assert result["place_name"] == "My Shop"

    def test_none_for_non_maps_url(self):
        assert parse_map_url("https://example.com") is None

    def test_none_for_empty_string(self):
        assert parse_map_url("") is None

    def test_none_for_none_input(self):
        assert parse_map_url(None) is None  # type: ignore[arg-type]

    def test_none_for_maps_url_without_coords(self):
        url = "https://www.google.com/maps/search/restaurants"
        assert parse_map_url(url) is None

    def test_auto_adds_https(self):
        url = "maps.google.com/maps?q=34.77,32.42"
        result = parse_map_url(url)
        assert result is not None
        assert result["lat"] == pytest.approx(34.77)

    def test_invalid_lat_range(self):
        url = "https://www.google.com/maps/@91.0,32.42,15z"
        assert parse_map_url(url) is None

    def test_invalid_lng_range(self):
        url = "https://www.google.com/maps/@34.77,181.0,15z"
        assert parse_map_url(url) is None

    def test_whitespace_stripped(self):
        url = "  https://www.google.com/maps/@34.77,32.42,15z  "
        result = parse_map_url(url)
        assert result is not None


class TestHaversineDistance:
    """Test haversine distance calculation."""

    def test_same_point(self):
        assert haversine_distance(34.77, 32.42, 34.77, 32.42) == 0.0

    def test_known_distance_paphos_limassol(self):
        # Paphos to Limassol: ~68 km
        dist = haversine_distance(34.7724, 32.4218, 34.6786, 33.0413)
        assert 55_000 < dist < 80_000  # rough range

    def test_known_distance_short(self):
        # Two points ~111 meters apart (0.001 degree lat at equator)
        dist = haversine_distance(0.0, 0.0, 0.001, 0.0)
        assert 100 < dist < 120

    def test_antipodal_points(self):
        # Maximum distance: ~20,000 km
        dist = haversine_distance(0, 0, 0, 180)
        assert 20_000_000 < dist < 20_100_000

    def test_symmetric(self):
        d1 = haversine_distance(34.77, 32.42, 35.18, 33.38)
        d2 = haversine_distance(35.18, 33.38, 34.77, 32.42)
        assert d1 == pytest.approx(d2)


class TestIsMapUrl:
    """Test quick map URL detection."""

    def test_full_google_maps_url(self):
        assert is_map_url("https://www.google.com/maps/place/Shop/@34.77,32.42")

    def test_maps_app_short_link(self):
        assert is_map_url("https://maps.app.goo.gl/abc123")

    def test_goo_gl_maps_link(self):
        assert is_map_url("https://goo.gl/maps/abc123")

    def test_maps_google_com(self):
        assert is_map_url("https://maps.google.com/maps?q=34.77,32.42")

    def test_non_maps_url(self):
        assert not is_map_url("https://example.com/page")

    def test_empty_string(self):
        assert not is_map_url("")

    def test_none_value(self):
        assert not is_map_url(None)  # type: ignore[arg-type]

    def test_whitespace_handling(self):
        assert is_map_url("  https://maps.app.goo.gl/abc  ")
