"""End-to-end RBAC leak test (ACL_PLAN.md §5), backend-agnostic.

Two documents name the SAME entity ``ACME`` with different access roles:

* Doc A (``allowed_roles=['public']``):     "ACME is a manufacturer ..." plus a
  public relation ``ACME -> PUMP_PRODUCT``.
* Doc B (``allowed_roles=['confidential']``): "ACME plans a secret takeover X"
  plus a confidential relation ``ACME -> TAKEOVER_TARGET``.

A ``public`` user must:
  (a) still SEE the entity ``ACME`` (it has an allowed source — Doc A),
  (b) get NO confidential description fragment ("takeover") on ACME,
  (c) get NO confidential relation / endpoint entity (``TAKEOVER_TARGET``),
  (d) get NO confidential chunk text ("takeover") in the mixed context.
A ``confidential`` user sees everything; a query WITHOUT ``user_roles`` (no
filter) also sees everything — backward compatible.

This runs entirely on the default backends (NetworkX graph + NanoVectorDB +
JSON-KV). Enforcement here is the backend-agnostic post-retrieval filtering;
Postgres additionally pre-filters in-DB (covered by the requires_db SQL tests).
The LLM/embedding are mocked so extraction is deterministic and every candidate
is retrieved (constant embedding => all pass the cosine gate), leaving the ACL
filter as the only thing that removes the confidential items.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface


class _SimpleTokenizerImpl(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _const_embedding(texts: list[str]) -> np.ndarray:
    # Identical vectors => cosine similarity 1.0 for every pair, so the vector
    # search returns every candidate regardless of text. The ACL filter is then
    # the sole gate on what reaches the context.
    return np.ones((len(texts), 32), dtype=np.float32)


_PUBLIC_EXTRACTION = (
    "entity<|#|>ACME<|#|>organization<|#|>ACME is a manufacturer of pumps.\n"
    "entity<|#|>PUMP_PRODUCT<|#|>product<|#|>A pump produced by ACME.\n"
    "relation<|#|>ACME<|#|>PUMP_PRODUCT<|#|>manufactures<|#|>ACME manufactures the pump product.\n"
    "<|COMPLETE|>"
)

_CONFIDENTIAL_EXTRACTION = (
    "entity<|#|>ACME<|#|>organization<|#|>ACME plans a secret takeover of X.\n"
    "entity<|#|>TAKEOVER_TARGET<|#|>organization<|#|>The secret takeover target X.\n"
    "relation<|#|>ACME<|#|>TAKEOVER_TARGET<|#|>acquires<|#|>ACME secretly plans to acquire the takeover target.\n"
    "<|COMPLETE|>"
)


async def _mock_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    text = f"{system_prompt or ''}\n{prompt}"
    # Keyword-extraction requests (should not happen — we pass keywords in the
    # QueryParam — but answer safely just in case).
    if "high_level_keywords" in text:
        return json.dumps(
            {"high_level_keywords": ["ACME"], "low_level_keywords": ["ACME"]}
        )
    if "takeover" in text:
        return _CONFIDENTIAL_EXTRACTION
    if "manufacturer" in text:
        return _PUBLIC_EXTRACTION
    return "<|COMPLETE|>"


def _new_rag(tmp_path: Path) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"acl-e2e-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32, max_token_size=4096, func=_const_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        entity_extract_max_gleaning=0,
        cosine_better_than_threshold=-1.0,  # accept every candidate
        chunk_token_size=200,
    )


async def _ingest(rag: LightRAG) -> None:
    await rag.ainsert(
        "ACME is a manufacturer of pumps. The pump product is widely used.",
        ids=["doc-public"],
        file_paths=["public.txt"],
        allowed_roles=["public"],
    )
    await rag.ainsert(
        "ACME plans a secret takeover of X. The takeover target is strategic.",
        ids=["doc-conf"],
        file_paths=["confidential.txt"],
        allowed_roles=["confidential"],
    )


def _query_data(rag: LightRAG, user_roles):
    param = QueryParam(
        mode="mix",
        user_roles=user_roles,
        ll_keywords=["ACME"],
        hl_keywords=["ACME"],
        top_k=100,
        chunk_top_k=100,
        enable_rerank=False,
    )
    return rag.aquery_data("Tell me about ACME", param)


def _flatten(data: dict) -> dict:
    payload = data.get("data", {}) or {}
    entities = payload.get("entities", []) or []
    relationships = payload.get("relationships", []) or []
    chunks = payload.get("chunks", []) or []
    entity_names = {e.get("entity_name") for e in entities}
    entity_descs = " ".join((e.get("description") or "") for e in entities)
    rel_blob = json.dumps(relationships)
    chunk_blob = " ".join((c.get("content") or "") for c in chunks)
    return {
        "entity_names": entity_names,
        "entity_descs": entity_descs,
        "rel_blob": rel_blob,
        "chunk_blob": chunk_blob,
        "raw": data,
    }


@pytest.mark.offline
def test_public_user_sees_no_confidential_leak(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await _ingest(rag)
            return (
                await _query_data(rag, ["public"]),
                await _query_data(rag, ["confidential"]),
                await _query_data(rag, None),
            )
        finally:
            await rag.finalize_storages()

    public_data, conf_data, open_data = asyncio.run(_run())
    pub = _flatten(public_data)
    conf = _flatten(conf_data)
    opn = _flatten(open_data)

    # (a) public user still sees the shared entity ACME
    assert "ACME" in pub["entity_names"], pub["raw"]
    # (b) ACME description carries the public fragment, not the confidential one
    assert "manufacturer" in pub["entity_descs"].lower()
    assert "takeover" not in pub["entity_descs"].lower()
    # (c) no confidential endpoint entity / relation
    assert "TAKEOVER_TARGET" not in pub["entity_names"]
    assert "takeover" not in pub["rel_blob"].lower()
    assert "TAKEOVER_TARGET" not in pub["rel_blob"]
    # (d) no confidential chunk text in the mixed context
    assert "takeover" not in pub["chunk_blob"].lower()

    # confidential user sees the confidential material
    assert "TAKEOVER_TARGET" in conf["entity_names"]
    assert "takeover" in (conf["entity_descs"] + conf["chunk_blob"]).lower()

    # no filter (backward compatible) sees everything too
    assert "TAKEOVER_TARGET" in opn["entity_names"]
    assert "takeover" in (opn["entity_descs"] + opn["chunk_blob"]).lower()


class _ForeignVectorStorage:
    """Stand-in for a backend that cannot enforce ACL (not in the allowlist)."""

    supports_acl_prefilter = False


@pytest.mark.offline
def test_user_roles_rejected_on_non_acl_backend(tmp_path):
    """A role-scoped query must hard-reject a non-ACL-capable backend."""

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            # Swap in a foreign, unverified vector backend.
            rag.entities_vdb = _ForeignVectorStorage()
            with pytest.raises(ValueError, match="not ACL-capable"):
                rag._enforce_acl_capability(QueryParam(user_roles=["public"]))
            # An open query (no user_roles) must still pass unchanged.
            rag._enforce_acl_capability(QueryParam())
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
