"""Document-based role-based access control (RBAC) — pure helpers.

See ``ACL_PLAN.md`` for the full design. This module holds only pure,
backend-agnostic helpers shared by the ingest merge path (``operate.py``)
and the query-time enforcement path. It performs **no** storage/IO.

Permission model (flat, OR semantics):
- An element (chunk / node / edge / description fragment) carries
  ``roles: list[str] | None``. ``None`` (or empty) means **open** — visible
  to everyone (backward compatible: data ingested before RBAC has no roles).
- A query carries ``user_roles: list[str] | None``. ``None`` means **no
  filter** — see everything (backward compatible: queries without roles).
- Visibility: an element is visible iff it is open, or the query has no
  filter, or ``intersection(element_roles, user_roles)`` is non-empty.

Provenance, not a single label: nodes/edges arise from multiple source
chunks. We keep a **per-source description fragment** (each with its own
ACL) and derive the node/edge ``roles`` as an *open-dominant union* of its
fragments. At query time the description is reassembled from **only** the
fragments the user is allowed to see — the mixed summary blob is never
emitted unfiltered.
"""

from __future__ import annotations

import json

from lightrag.constants import GRAPH_FIELD_SEP

# Sentinel source_id for the implicit fragment that represents a pre-RBAC
# node/edge: legacy data carried a single (open) description blob and no
# per-source fragments. Treating it as one open fragment keeps previously
# visible data visible after the feature lands.
LEGACY_FRAGMENT_SOURCE = "<legacy>"


def normalize_roles(value) -> list[str] | None:
    """Coerce ``value`` into an order-preserving, de-duplicated ``list[str]``.

    Returns ``None`` for an absent/empty value (the *open* state). Accepts a
    single string, or any iterable of strings. Non-string members are
    stringified and stripped; blanks are dropped.
    """
    if value is None:
        return None
    if isinstance(value, str):
        items: list = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        return None

    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out or None


def roles_overlap(element_roles, user_roles) -> bool:
    """OR-visibility check.

    Visible iff the query has no filter (``user_roles is None``), or the
    element is open (``element_roles`` empty/None), or the two sets intersect.
    """
    if user_roles is None:
        return True
    er = normalize_roles(element_roles)
    if er is None:
        return True  # open element — visible to everyone
    ur = normalize_roles(user_roles)
    if ur is None:
        return True
    ur_set = set(ur)
    return any(role in ur_set for role in er)


def encode_roles_for_graph(roles) -> str:
    """Serialize a roles list to a graph node/edge string property.

    Returns ``""`` for the open state. We use a ``GRAPH_FIELD_SEP``-joined
    string (same convention as ``source_id``/``file_path``) so the value
    round-trips identically on NetworkX *and* AGE/Postgres graph backends.
    """
    normalized = normalize_roles(roles)
    return GRAPH_FIELD_SEP.join(normalized) if normalized else ""


def decode_roles_from_graph(value) -> list[str] | None:
    """Parse a graph ``roles`` property back to ``list[str] | None`` (open)."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        return normalize_roles(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return normalize_roles(stripped.split(GRAPH_FIELD_SEP))
    return None


def encode_fragments(fragments) -> str:
    """Serialize the per-source description fragments to a JSON string.

    Stored as a single graph node/edge property (``desc_fragments``); empty
    list serializes to ``""`` so legacy/open elements add no property bloat.
    """
    if not fragments:
        return ""
    return json.dumps(fragments, ensure_ascii=False)


def decode_fragments(value) -> list[dict]:
    """Parse the ``desc_fragments`` property back to a list of fragment dicts."""
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def union_roles_from_fragments(fragments) -> list[str] | None:
    """Open-dominant union of fragment roles.

    If **any** fragment is open (no roles), the whole element is open
    (returns ``None``) — that source is visible to everyone, so the element
    is reachable by everyone. Otherwise returns the de-duplicated union of
    every fragment's roles.
    """
    if not fragments:
        return None
    union: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        fr_roles = normalize_roles(fragment.get("roles"))
        if fr_roles is None:
            return None  # an open source makes the whole element open
        for role in fr_roles:
            if role not in seen:
                seen.add(role)
                union.append(role)
    return union or None


def _coerce_fragment(fragment: dict) -> dict | None:
    source_id = fragment.get("source_id")
    if source_id is None:
        return None
    return {
        "source_id": source_id,
        "description": fragment.get("description", "") or "",
        "roles": normalize_roles(fragment.get("roles")),
    }


def merge_fragments(existing, new_fragments) -> list[dict]:
    """Merge per-source description fragments, keyed by ``source_id``.

    Provenance only ever grows or updates in place: a fragment for a given
    source replaces an earlier one for the same source (handles re-ingest),
    new sources are appended. Insertion order is preserved for stable output.
    """
    by_source: dict[str, dict] = {}
    order: list[str] = []
    for fragment in list(existing or []) + list(new_fragments or []):
        coerced = _coerce_fragment(fragment)
        if coerced is None:
            continue
        sid = coerced["source_id"]
        if sid not in by_source:
            order.append(sid)
        by_source[sid] = coerced
    return [by_source[sid] for sid in order]


def assemble_description(
    fragments,
    user_roles,
    fallback: str = "",
    separator: str = GRAPH_FIELD_SEP,
) -> str:
    """Rebuild a description from only the fragments visible to ``user_roles``.

    When ``user_roles is None`` (no filter) every fragment is included. The
    confidential text of a forbidden fragment is never concatenated — this is
    the anti-leak core. Returns ``fallback`` when nothing is visible.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for fragment in fragments or []:
        if not roles_overlap(fragment.get("roles"), user_roles):
            continue
        desc = (fragment.get("description") or "").strip()
        if desc and desc not in seen:
            seen.add(desc)
            parts.append(desc)
    if not parts:
        return fallback
    return separator.join(parts)
