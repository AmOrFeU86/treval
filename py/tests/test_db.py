"""Tests for SpanStore.

Focused on bugs in the public API of treval's local SQLite store.
"""
import json

from treval.db import SpanStore


def test_update_json_encodes_metadata_dict(tmp_path):
    """SpanStore.update() must JSON-encode a dict metadata, like save() does.

    Regression: previously only save() encoded metadata via json.dumps,
    so update() storing a dict crashed with
    "sqlite3.ProgrammingError: Error binding parameter: type 'dict' is
    not supported". This test pins the contract: a dict passed via
    update() is encoded just like a dict passed via save(), so callers
    can use the same shape regardless of whether they save or update.
    """
    db = tmp_path / "spans.db"
    store = SpanStore(db_path=db)
    span_id = store.save(name="s1", type="OPERATION")

    store.update(span_id, metadata={"k": "v", "n": 42})

    row = store.get(span_id)
    assert row["metadata"] is not None
    # Stored as a JSON string in the column
    decoded = json.loads(row["metadata"])
    assert decoded == {"k": "v", "n": 42}


def test_update_passes_through_string_metadata_unchanged(tmp_path):
    """If the caller pre-encodes metadata as a string, update() does not
    double-encode. This matches save()'s tolerance — `save` only calls
    `json.dumps` if metadata is not None and not already a string.
    """
    db = tmp_path / "spans.db"
    store = SpanStore(db_path=db)
    span_id = store.save(name="s1", type="OPERATION")

    pre_encoded = json.dumps({"k": "v"})
    store.update(span_id, metadata=pre_encoded)

    row = store.get(span_id)
    assert row["metadata"] == pre_encoded


def test_update_metadata_none_clears_field(tmp_path):
    """Passing metadata=None (the default when the field is omitted)
    must not touch the existing metadata in the column.
    """
    db = tmp_path / "spans.db"
    store = SpanStore(db_path=db)
    span_id = store.save(name="s1", type="OPERATION",
                         metadata={"initial": True})

    # Update other fields, leave metadata alone
    store.update(span_id, status="error")

    row = store.get(span_id)
    assert row["status"] == "error"
    assert json.loads(row["metadata"]) == {"initial": True}
