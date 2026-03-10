"""Tests for alibi.auth.keys — mnemonic generation, hashing, prefix."""

from alibi.auth.keys import generate_mnemonic, hash_key, key_prefix


class TestGenerateMnemonic:
    def test_default_word_count(self):
        mnemonic = generate_mnemonic()
        words = mnemonic.split()
        assert len(words) == 6

    def test_custom_word_count(self):
        mnemonic = generate_mnemonic(word_count=8)
        words = mnemonic.split()
        assert len(words) == 8

    def test_words_are_lowercase(self):
        mnemonic = generate_mnemonic()
        assert mnemonic == mnemonic.lower()

    def test_different_each_time(self):
        m1 = generate_mnemonic()
        m2 = generate_mnemonic()
        # With 2048^6 possibilities, collision is astronomically unlikely
        assert m1 != m2

    def test_words_from_bip39(self):
        from alibi.auth.keys import _load_wordlist

        wordlist = set(_load_wordlist())
        mnemonic = generate_mnemonic()
        for word in mnemonic.split():
            assert word in wordlist


class TestHashKey:
    def test_deterministic(self):
        h1 = hash_key("abandon ability able about above absent")
        h2 = hash_key("abandon ability able about above absent")
        assert h1 == h2

    def test_hex_string(self):
        h = hash_key("abandon ability")
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)

    def test_case_insensitive(self):
        h1 = hash_key("Abandon Ability")
        h2 = hash_key("abandon ability")
        assert h1 == h2

    def test_normalizes_whitespace(self):
        h1 = hash_key("abandon  ability   able")
        h2 = hash_key("abandon ability able")
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        h1 = hash_key("abandon ability")
        h2 = hash_key("ability abandon")
        assert h1 != h2


class TestKeyPrefix:
    def test_extracts_first_two_words(self):
        prefix = key_prefix("abandon ability able about above absent")
        assert prefix == "abandon ability"

    def test_lowercases(self):
        prefix = key_prefix("Abandon Ability Able")
        assert prefix == "abandon ability"

    def test_single_word(self):
        prefix = key_prefix("abandon")
        assert prefix == "abandon"
