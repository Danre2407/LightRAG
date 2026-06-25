"""Unit tests for the pure RBAC helpers in ``lightrag/acl.py`` (S3/S4/S7).

These lock the access-control *logic* independently of any storage backend:
role normalization, OR-visibility, the open-dominant roles union, per-source
fragment merging, and the anti-leak description assembly.
"""

from __future__ import annotations

import pytest

from lightrag.acl import (
    LEGACY_FRAGMENT_SOURCE,
    assemble_description,
    decode_fragments,
    decode_roles_from_graph,
    encode_fragments,
    encode_roles_for_graph,
    merge_fragments,
    normalize_roles,
    roles_overlap,
    union_roles_from_fragments,
)
from lightrag.constants import GRAPH_FIELD_SEP


# --------------------------------------------------------------------------- #
# normalize_roles
# --------------------------------------------------------------------------- #


@pytest.mark.offline
def test_normalize_roles_variants():
    assert normalize_roles(None) is None
    assert normalize_roles([]) is None
    assert normalize_roles("") is None
    assert normalize_roles("public") == ["public"]
    assert normalize_roles(["public", "public", " conf "]) == ["public", "conf"]
    assert normalize_roles(("a", "b", "a")) == ["a", "b"]
    # Non-iterable / unsupported -> open
    assert normalize_roles(123) is None


# --------------------------------------------------------------------------- #
# roles_overlap (OR visibility)
# --------------------------------------------------------------------------- #


@pytest.mark.offline
def test_roles_overlap_semantics():
    # No filter -> everything visible
    assert roles_overlap(["confidential"], None) is True
    # Open element -> visible to everyone
    assert roles_overlap(None, ["public"]) is True
    assert roles_overlap([], ["public"]) is True
    # Intersection non-empty
    assert roles_overlap(["public", "x"], ["public"]) is True
    # Disjoint -> hidden
    assert roles_overlap(["confidential"], ["public"]) is False


# --------------------------------------------------------------------------- #
# graph encode/decode round-trip
# --------------------------------------------------------------------------- #


@pytest.mark.offline
def test_roles_graph_roundtrip():
    assert encode_roles_for_graph(None) == ""
    assert encode_roles_for_graph([]) == ""
    encoded = encode_roles_for_graph(["public", "confidential"])
    assert encoded == f"public{GRAPH_FIELD_SEP}confidential"
    assert decode_roles_from_graph(encoded) == ["public", "confidential"]
    assert decode_roles_from_graph("") is None
    assert decode_roles_from_graph(None) is None
    assert decode_roles_from_graph(["a", "b"]) == ["a", "b"]


@pytest.mark.offline
def test_fragments_roundtrip():
    frags = [{"source_id": "c1", "description": "d", "roles": ["public"]}]
    assert decode_fragments(encode_fragments(frags)) == frags
    assert encode_fragments([]) == ""
    assert decode_fragments("") == []
    assert decode_fragments(None) == []
    assert decode_fragments("not json") == []


# --------------------------------------------------------------------------- #
# union_roles_from_fragments (open-dominant)
# --------------------------------------------------------------------------- #


@pytest.mark.offline
def test_union_roles_open_dominant():
    # All restricted -> union
    frags = [
        {"source_id": "c1", "roles": ["public"]},
        {"source_id": "c2", "roles": ["confidential"]},
    ]
    assert union_roles_from_fragments(frags) == ["public", "confidential"]
    # Any open fragment -> whole element open (None)
    frags_with_open = frags + [{"source_id": "c3", "roles": None}]
    assert union_roles_from_fragments(frags_with_open) is None
    assert union_roles_from_fragments([]) is None


# --------------------------------------------------------------------------- #
# merge_fragments (provenance grows / updates per source)
# --------------------------------------------------------------------------- #


@pytest.mark.offline
def test_merge_fragments_keyed_by_source():
    existing = [{"source_id": "c1", "description": "old", "roles": ["public"]}]
    new = [
        {"source_id": "c1", "description": "updated", "roles": ["public"]},
        {"source_id": "c2", "description": "secret", "roles": ["confidential"]},
    ]
    merged = merge_fragments(existing, new)
    assert [f["source_id"] for f in merged] == ["c1", "c2"]
    assert merged[0]["description"] == "updated"  # same source -> replaced
    assert merged[1]["roles"] == ["confidential"]


# --------------------------------------------------------------------------- #
# assemble_description (anti-leak core)
# --------------------------------------------------------------------------- #


@pytest.mark.offline
def test_assemble_description_filters_forbidden_fragments():
    fragments = [
        {"source_id": "c1", "description": "ACME is a manufacturer.", "roles": ["public"]},
        {"source_id": "c2", "description": "ACME plans takeover X.", "roles": ["confidential"]},
    ]
    # public user: only the public fragment, never the confidential text
    public_desc = assemble_description(fragments, ["public"])
    assert "manufacturer" in public_desc
    assert "takeover" not in public_desc
    # confidential user: sees both
    conf_desc = assemble_description(fragments, ["confidential", "public"])
    assert "manufacturer" in conf_desc and "takeover" in conf_desc
    # no filter -> everything
    assert "takeover" in assemble_description(fragments, None)


@pytest.mark.offline
def test_assemble_description_fallback_when_nothing_visible():
    fragments = [
        {"source_id": "c2", "description": "secret", "roles": ["confidential"]},
    ]
    assert assemble_description(fragments, ["public"], fallback="Entity ACME") == "Entity ACME"


@pytest.mark.offline
def test_legacy_fragment_sentinel_is_open():
    # A pre-RBAC element is modeled as one open fragment -> stays visible and
    # its text is shown to everyone.
    fragments = [
        {"source_id": LEGACY_FRAGMENT_SOURCE, "description": "legacy blob", "roles": None},
        {"source_id": "c2", "description": "secret", "roles": ["confidential"]},
    ]
    assert union_roles_from_fragments(fragments) is None  # open
    public_desc = assemble_description(fragments, ["public"])
    assert public_desc == "legacy blob"
