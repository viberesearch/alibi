"""Mnemonic API key generation using BIP39 English wordlist.

Generates 6-word passphrases (~66 bits entropy) as human-friendly API keys.
Keys are stored as PBKDF2-SHA256 hashes with per-key random salts.
The plaintext is shown once at creation.
"""

import hashlib
import secrets
from pathlib import Path

_WORDLIST_PATH = Path(__file__).parent / "bip39_english.txt"
_wordlist: list[str] | None = None

_PBKDF2_ITERATIONS = 100_000


def _load_wordlist() -> list[str]:
    """Load and cache the BIP39 English wordlist (2048 words)."""
    global _wordlist
    if _wordlist is None:
        _wordlist = _WORDLIST_PATH.read_text().strip().splitlines()
        if len(_wordlist) != 2048:
            raise ValueError(
                f"Expected 2048 words in BIP39 wordlist, got {len(_wordlist)}"
            )
    return _wordlist


def generate_mnemonic(word_count: int = 6) -> str:
    """Generate a random mnemonic passphrase.

    Args:
        word_count: Number of words (default 6, ~66 bits entropy).

    Returns:
        Space-separated lowercase words.
    """
    words = _load_wordlist()
    chosen = [secrets.choice(words) for _ in range(word_count)]
    return " ".join(chosen)


def generate_salt() -> bytes:
    """Generate a 16-byte random salt for PBKDF2 key hashing."""
    return secrets.token_bytes(16)


def hash_key(mnemonic: str, salt: bytes | None = None) -> str:
    """Hash a mnemonic passphrase for storage using PBKDF2-SHA256.

    Args:
        mnemonic: The plaintext mnemonic to hash.
        salt: Per-key random salt. If None, falls back to unsalted SHA-256
              for backward compatibility with legacy keys.

    Returns:
        Hex-encoded digest.
    """
    normalized = " ".join(mnemonic.lower().split())
    if salt is None:
        # Legacy unsalted SHA-256 (for validating old keys during migration)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return hashlib.pbkdf2_hmac(
        "sha256",
        normalized.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
    ).hex()


def key_prefix(mnemonic: str) -> str:
    """Extract the first 2 words as a log-safe display prefix.

    Returns:
        First two words separated by a space, e.g. "abandon ability".
    """
    words = mnemonic.lower().split()
    return " ".join(words[:2])
