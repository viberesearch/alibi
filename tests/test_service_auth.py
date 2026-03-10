"""Tests for alibi.services.auth — user CRUD, API keys, contacts."""

from alibi.services.auth import (
    add_contact,
    create_api_key,
    create_user,
    find_user_by_contact,
    find_user_by_telegram,
    get_display_name,
    get_user,
    link_telegram,
    list_api_keys,
    list_contacts,
    list_users,
    remove_contact,
    remove_contact_by_value,
    revoke_api_key,
    update_user,
    validate_api_key,
)


class TestCreateUser:
    def test_creates_user_no_name(self, db):
        result = create_user(db)
        assert result["name"] is None
        assert result["id"]
        assert result["is_active"] == 1

    def test_creates_user_with_name(self, db):
        result = create_user(db, "Alice")
        assert result["name"] == "Alice"


class TestUpdateUser:
    def test_updates_name(self, db):
        user = create_user(db)
        assert update_user(db, user["id"], name="Bob")
        updated = get_user(db, user["id"])
        assert updated["name"] == "Bob"

    def test_clears_name(self, db):
        user = create_user(db, "Alice")
        assert update_user(db, user["id"], name=None)
        updated = get_user(db, user["id"])
        assert updated["name"] is None

    def test_returns_false_for_unknown(self, db):
        assert not update_user(db, "nonexistent", name="X")


class TestGetUser:
    def test_returns_user(self, db):
        created = create_user(db, "Alice")
        user = get_user(db, created["id"])
        assert user is not None
        assert user["name"] == "Alice"

    def test_returns_none_for_unknown(self, db):
        assert get_user(db, "nonexistent") is None


class TestDisplayName:
    def test_returns_name_when_set(self, db):
        assert get_display_name({"name": "Alice"}) == "Alice"

    def test_returns_user_when_none(self, db):
        assert get_display_name({"name": None}) == "user"

    def test_returns_user_when_empty(self, db):
        assert get_display_name({"name": ""}) == "user"


class TestListUsers:
    def test_lists_users(self, db):
        create_user(db, "Alice")
        create_user(db, "Bob")
        users = list_users(db)
        names = {u["name"] for u in users}
        assert "Alice" in names
        assert "Bob" in names


class TestContacts:
    def test_add_telegram_contact(self, db):
        user = create_user(db)
        result = add_contact(db, user["id"], "telegram", "12345")
        assert result is not None
        assert result["contact_type"] == "telegram"
        assert result["value"] == "12345"

    def test_add_email_contact(self, db):
        user = create_user(db)
        result = add_contact(db, user["id"], "email", "alice@example.com")
        assert result is not None
        assert result["contact_type"] == "email"

    def test_add_contact_with_label(self, db):
        user = create_user(db)
        result = add_contact(db, user["id"], "telegram", "12345", label="personal")
        assert result["label"] == "personal"

    def test_add_contact_returns_none_for_unknown_user(self, db):
        assert add_contact(db, "nonexistent", "telegram", "12345") is None

    def test_multiple_telegrams_per_user(self, db):
        user = create_user(db)
        add_contact(db, user["id"], "telegram", "111")
        add_contact(db, user["id"], "telegram", "222")
        contacts = list_contacts(db, user["id"])
        tg_values = {c["value"] for c in contacts if c["contact_type"] == "telegram"}
        assert tg_values == {"111", "222"}

    def test_multiple_emails_per_user(self, db):
        user = create_user(db)
        add_contact(db, user["id"], "email", "a@x.com")
        add_contact(db, user["id"], "email", "b@x.com")
        contacts = list_contacts(db, user["id"])
        assert len(contacts) == 2

    def test_same_telegram_cannot_link_twice(self, db):
        import sqlite3

        user1 = create_user(db)
        user2 = create_user(db)
        add_contact(db, user1["id"], "telegram", "12345")
        try:
            add_contact(db, user2["id"], "telegram", "12345")
            assert False, "Should have raised IntegrityError"
        except sqlite3.IntegrityError:
            pass

    def test_remove_contact_by_id(self, db):
        user = create_user(db)
        result = add_contact(db, user["id"], "telegram", "12345")
        assert remove_contact(db, result["id"])
        assert list_contacts(db, user["id"]) == []

    def test_remove_contact_by_value(self, db):
        user = create_user(db)
        add_contact(db, user["id"], "telegram", "12345")
        assert remove_contact_by_value(db, "telegram", "12345")
        assert list_contacts(db, user["id"]) == []

    def test_remove_nonexistent_returns_false(self, db):
        assert not remove_contact(db, "nonexistent")

    def test_find_user_by_contact(self, db):
        user = create_user(db, "Alice")
        add_contact(db, user["id"], "telegram", "12345")
        found = find_user_by_contact(db, "telegram", "12345")
        assert found is not None
        assert found["name"] == "Alice"

    def test_find_user_by_contact_not_found(self, db):
        assert find_user_by_contact(db, "telegram", "99999") is None

    def test_find_user_by_telegram_convenience(self, db):
        user = create_user(db, "Alice")
        add_contact(db, user["id"], "telegram", "12345")
        found = find_user_by_telegram(db, "12345")
        assert found is not None
        assert found["name"] == "Alice"


class TestLinkTelegramCompat:
    """Backward compat wrapper tests."""

    def test_links_telegram(self, db):
        user = create_user(db, "Alice")
        assert link_telegram(db, user["id"], "12345")
        found = find_user_by_telegram(db, "12345")
        assert found is not None

    def test_returns_false_for_unknown_user(self, db):
        assert not link_telegram(db, "nonexistent", "12345")


class TestApiKeys:
    def test_create_and_validate(self, db):
        user = create_user(db, "Alice")
        key_result = create_api_key(db, user["id"], label="test")

        assert "mnemonic" in key_result
        assert key_result["prefix"]
        assert key_result["label"] == "test"

        # Validate the mnemonic
        validated = validate_api_key(db, key_result["mnemonic"])
        assert validated is not None
        assert validated["id"] == user["id"]
        assert validated["name"] == "Alice"

    def test_validate_returns_none_name_for_unnamed_user(self, db):
        user = create_user(db)
        key_result = create_api_key(db, user["id"])
        validated = validate_api_key(db, key_result["mnemonic"])
        assert validated["name"] is None

    def test_invalid_key_returns_none(self, db):
        assert validate_api_key(db, "wrong key words here now please") is None

    def test_list_keys(self, db):
        user = create_user(db, "Alice")
        create_api_key(db, user["id"], label="k1")
        create_api_key(db, user["id"], label="k2")

        keys = list_api_keys(db, user["id"])
        assert len(keys) == 2
        labels = {k["label"] for k in keys}
        assert labels == {"k1", "k2"}
        # No plaintext mnemonic in list
        for k in keys:
            assert "mnemonic" not in k

    def test_revoke_key(self, db):
        user = create_user(db, "Alice")
        key_result = create_api_key(db, user["id"])

        assert revoke_api_key(db, key_result["id"])

        # Key should no longer validate
        assert validate_api_key(db, key_result["mnemonic"]) is None

    def test_revoke_nonexistent_returns_false(self, db):
        assert not revoke_api_key(db, "nonexistent")

    def test_validates_updates_last_used(self, db):
        user = create_user(db, "Alice")
        key_result = create_api_key(db, user["id"])

        # Before validation, last_used_at should be None
        keys = list_api_keys(db, user["id"])
        assert keys[0]["last_used_at"] is None

        # Validate
        validate_api_key(db, key_result["mnemonic"])

        # After validation, last_used_at should be set
        keys = list_api_keys(db, user["id"])
        assert keys[0]["last_used_at"] is not None
