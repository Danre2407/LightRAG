# ACL_PLAN.md — Rollenbasierte Zugriffskontrolle (RBAC) auf gemeinsamem Knowledge Graph

> Status: **Schritt 1 — Erkundung & Plan. Noch kein Code geändert.**
> Backend-Ziel: **Postgres + pgvector + AGE** (`lightrag/kg/postgres_impl.py`).
> Die file-basierten Defaults (NanoVectorDB / NetworkX) bleiben unangetastet.

> **Getroffene Design-Entscheidungen:**
> 1. **Beschreibung bei aktivem ACL immer aus Fragmenten neu bauen** — bei gesetztem
>    `user_roles` wird die Beschreibung ausschließlich aus erlaubten
>    `LIGHTRAG_DESC_FRAGMENTS` zusammengesetzt; der gemischte Summary-Blob geht
>    **nie** ungefiltert in den Kontext.
> 2. **Fremd-Backends hart ablehnen** — eine Query mit `user_roles` gegen ein
>    nicht-ACL-fähiges Backend (NanoVectorDB/NetworkX/Neo4j/Milvus/Qdrant/…) wirft
>    einen Fehler statt still ungefiltert zu laufen.

---

## 1. Architektur-Erkundung (verifiziert am Code)

### 1.1 INGEST-Pipeline: Dokument → Chunk → Entity/Relation → Merge → Storage

| Schritt | Datei / Funktion | Was passiert | ACL-Relevanz |
|---|---|---|---|
| Einstieg SDK | `lightrag/lightrag.py::ainsert` (1428) → `apipeline_enqueue_documents` → `apipeline_process_enqueue_documents` | Nimmt `input`, `ids`, `file_paths`, `track_id`. **Kein `metadata`/`allowed_roles`-Parameter heute.** | **Einschleusepunkt #1**: neuer Parameter `allowed_roles`. |
| Enqueue / Persistenz | `lightrag/pipeline.py::apipeline_enqueue_documents` (~241) → schreibt `full_docs` + `doc_status` | `doc_status` hat bereits `metadata` JSONB (`base["metadata"]`, pipeline.py:599-609). `full_docs` (Tabelle `LIGHTRAG_DOC_FULL`) hat `meta JSONB`. | `allowed_roles` hier provenienz-stabil **pro Dokument** ablegen. |
| Chunking | `pipeline.py::_process_single_document` (~2043), Strategien F/R/V/P (2188-2335) → `build_chunks_dict_from_chunking_result` (`utils_pipeline.py:43`) | Baut `chunks: {chunk_key: {content, full_doc_id, file_path, tokens, ...}}`. `full_doc_id = doc_id`. | **Hier `roles` pro Chunk stempeln** (aus den Doc-Roles, die via `content_data`/`full_docs` verfügbar sind). |
| Chunk-Storage | `pipeline.py:2501-2502`: `self.chunks_vdb.upsert(chunks)` + `self.text_chunks.upsert(chunks)` | Schreibt nach `LIGHTRAG_VDB_CHUNKS` (Vektor) und `LIGHTRAG_DOC_CHUNKS` (KV). | **Beide Tabellen brauchen `roles`-Spalte.** |
| Extraktion | `pipeline.py:2524 _process_extract_entities` → `operate.py::extract_entities` (3320) | Iteriert Chunks, LLM-Extraktion. Records via `_handle_single_entity_extraction` (502) / `_handle_single_relationship_extraction` (589). | Jeder Record bekommt `source_id=chunk_key`, `file_path`. **`roles` aus dem Chunk in den Record durchreichen.** |
| **Merge (Kernpunkt!)** | `operate.py::merge_nodes_and_edges` (2914) → `_merge_nodes_then_upsert` (2000) / `_merge_edges_then_upsert` (2329) | Sammelt `already_*` aus Graph + neue Records. `source_id = GRAPH_FIELD_SEP.join(source_ids)` (Chunk-Liste, 2130). **`description_list = already_description + sorted_descriptions` (2159) → `_handle_entity_relation_summary` (265) macht daraus EINEN LLM-Blob (2176).** | **HIER ist die Leak-Quelle.** Beschreibung wird über alle Quellen/Rollen zu einem Mischblob. |
| Graph-Upsert | `_merge_nodes_then_upsert:2283-2295` `upsert_node` mit `{entity_type, description, source_id, file_path}`; VDB-Upsert 2297-2319 (`content = entity_name\ndescription`) | Node-Properties + Entity-VDB-Row. Edge analog (`_merge_edges_then_upsert`, ~2560+). | Node/Edge brauchen `roles` (Union der Quellen) **plus** rollen-getaggte Beschreibungs-Fragmente. |
| Zusatz-Provenienz | `entity_chunks_storage` / `relation_chunks_storage` (KV, Tabellen `LIGHTRAG_ENTITY_CHUNKS`/`LIGHTRAG_RELATION_CHUNKS`) | Halten die **vollständige** Chunk-ID-Liste pro Entity/Relation (über `source_id`-Limit hinaus). | Nützlich, aber Chunk→Roles-Lookup wäre teuer; wir taggen Roles direkt. |

**Wichtige Erkenntnis zur Provenienz:** `source_id` ist **bereits** eine `GRAPH_FIELD_SEP`-getrennte Liste von Chunk-IDs — Provenienz existiert also schon auf Quell-Ebene. Was fehlt: (a) Chunk→Roles, (b) rollen-getrennte Beschreibungs-Fragmente statt LLM-Mischblob.

`GRAPH_FIELD_SEP` = `<SEP>` (in `lightrag/constants.py`).

### 1.2 QUERY-Pipeline: Retrieval-Modi, Kontextaufbau, Store-Abfragen

| Modus | Pfad (`operate.py`) | Stores abgefragt |
|---|---|---|
| `naive` | `naive_query` (5715) → `_get_vector_context` (4258) | `chunks_vdb.query` |
| `local` | `_perform_kg_search` (4315) → `_get_node_data` (5144) | `entities_vdb.query` → Graph `get_nodes_batch` → `_find_most_related_edges_from_entities` (5204) |
| `global` | `_get_edge_data` (5419) | `relationships_vdb.query` → Graph `get_edges_batch` → `_find_most_related_entities_from_relationships` (5478) |
| `hybrid` | local + global | beide |
| `mix` | hybrid + `_get_vector_context` (4437) | + `chunks_vdb.query` |

**Gemeinsamer Kontextaufbau:** `kg_query` (3786) → `_build_query_context` (5024) → `_perform_kg_search` + `_apply_token_truncation` (4525) + `_build_context_str` (4839). Text-Chunks für KG-Treffer: `_find_related_text_unit_from_entities` (5260) und `_find_related_text_unit_from_relations` (5511) — beide expandieren `entity["source_id"]` zu Chunk-IDs (`split_string_by_multi_markers`, 5286) und holen Inhalte via `text_chunks_db.get_by_ids` (5397).

**Postgres-Vektorabfrage** (`postgres_impl.py::PGVectorStorage.query`, 4257): nutzt `SQL_TEMPLATES[self.namespace]` (`"chunks"`/`"entities"`/`"relationships"`, 8362-8391) mit Positionsparametern `[workspace, threshold, top_k, embedding]`. **Hier muss der Pre-Filter rein.**

**Die drei Durchsetzungspunkte (aus dem Code):**
1. **Vektorsuche (Pre-Filter, DB):** `entities`/`relationships`/`chunks`-SQL-Templates → `WHERE`-Klausel mit `roles && $5`.
2. **Graph-Traversierung:** `_find_most_related_edges_from_entities` (5204), `_get_node_data`/`_get_edge_data` (Nodes/Edges aus Graph) — verbotene Nodes/Edges überspringen (Roles als Graph-Property prüfen).
3. **Kontext-/Beschreibungsaufbau:** `_find_related_text_unit_from_entities`/`_from_relations` (Chunk-Filter) **und** Beschreibungs-Assembly aus erlaubten Fragmenten in `_build_context_str` (4839).

**QueryParam-Durchreichung:** `QueryParam` (`base.py:82`) → `aquery` (lightrag.py) → `kg_query`/`naive_query` → `_perform_kg_search` → `*.query`. Neuer Parameter `user_roles` wandert genau diesen Pfad. Heute hat `BaseVectorStorage.query` (`base.py:267`) **keine** Roles-Parameter → Signatur muss erweitert werden (mit Default `None` = rückwärtskompatibel).

### 1.3 Postgres-Schema (heute, `postgres_impl.py::TABLES`)

- `LIGHTRAG_DOC_CHUNKS` (7987): KV-Chunks, **keine** Roles.
- `LIGHTRAG_VDB_CHUNKS` (8004): Vektor-Chunks, `content_vector`, **keine** Roles.
- `LIGHTRAG_VDB_ENTITY` (8019): `entity_name`, `content`, `content_vector`, `chunk_ids varchar[]`.
- `LIGHTRAG_VDB_RELATION` (8033): `source_id`, `target_id`, `content`, `content_vector`, `chunk_ids varchar[]`.
- Graph: AGE (`PGGraphStorage`, 5925), Node/Edge-Properties als agtype. `upsert_node` (6485), `upsert_edge` (6541).

---

## 2. Berechtigungsmodell (wie gefordert)

- Rollen = Strings. Dokument bringt `allowed_roles: [..]`. Query bringt `user_roles: [..]` (über `QueryParam`).
- **Sichtbarkeit (OR):** Element sichtbar ⇔ `Schnittmenge(element_roles, user_roles) ≠ ∅`.
- **Provenienz statt Einzel-Label:** ACL **pro Quelle** (`source_id`/Chunk). Node/Edge sichtbar ⇔ User hat ≥1 erlaubte Quelle. Inkrementelle Updates: ACL-Menge nur **ergänzen** (Union).
- **Beschreibung getrennt halten:** Pro Quelle/ACL ein Beschreibungs-Fragment mit eigener ACL. Zur Query-Zeit nur erlaubte Fragmente zusammensetzen. **Kein** LLM-Summary-Mischblob über Rollengrenzen.
- **Rückwärtskompatibel / „open“-Semantik:**
  - Insert ohne `allowed_roles` → `roles = NULL` (offen, für alle sichtbar).
  - Query ohne `user_roles` → kein Filter (sieht alles, wie heute).
  - SQL-Prädikat: `($N::varchar[] IS NULL OR roles IS NULL OR roles && $N::varchar[])`.

---

## 3. Datenmodell-Änderungen (Postgres)

**Neue Spalten (`roles varchar[] NULL`, GIN-indiziert):**
- `LIGHTRAG_DOC_CHUNKS.roles`, `LIGHTRAG_VDB_CHUNKS.roles`
- `LIGHTRAG_VDB_ENTITY.roles` (Union aller Quell-Fragment-Rollen)
- `LIGHTRAG_VDB_RELATION.roles` (Union)
- Graph-Node/-Edge: Property `roles` (Union) **für Traversierungs-Filter**.

**Beschreibungs-Fragmente (Anti-Leak-Kern) — neue Tabelle:**
```
LIGHTRAG_DESC_FRAGMENTS (
    workspace varchar, element_type varchar('entity'|'relation'),
    element_id text,            -- entity_name  bzw.  src<SEP>tgt
    source_id varchar,          -- Chunk-ID (Provenienz)
    description text,
    roles varchar[] NULL,
    create_time timestamp,
    PRIMARY KEY (workspace, element_type, element_id, source_id)
)
```
- Beim Merge: **pro Quelle ein Fragment** schreiben (statt nur Summary-Blob).
- Node-`description`-Blob bleibt als Anzeige/Embedding-Quelle erhalten, wird aber **zur Query-Zeit nicht** ungefiltert ausgegeben, wenn `user_roles` gesetzt ist — dann wird die Beschreibung aus erlaubten Fragmenten neu zusammengesetzt.

**Migration:** additive `ALTER TABLE ... ADD COLUMN IF NOT EXISTS roles varchar[]` + `CREATE TABLE IF NOT EXISTS LIGHTRAG_DESC_FRAGMENTS` + GIN-Indizes, eingehängt in den bestehenden Migrations-/DDL-Mechanismus (`postgres_impl.py` ~1765 `ddl`-Ausführung, `_migrate_*`). Bestehende Zeilen: `roles = NULL` → bleiben offen (kompatibel).

---

## 4. Inkrementeller Umsetzungsplan (kleine Schritte, je einzeln testbar)

> **Stopp nach diesem Dokument für dein OK. Erst dann Schritt S1.**

- **S0 — Test-Harness/Fixtures:** Postgres-Marker (`requires_db`) sichten (`tests/kg/postgres_impl/`), E2E-Leak-Testgerüst skizzieren (noch rot/skipped).
- **S1 — API-Durchreichung Insert: ✅ ERLEDIGT.** `allowed_roles` in `ainsert`/`insert` + `apipeline_enqueue_documents`; normalisiert via neuer Helper `normalize_allowed_roles` (`utils_pipeline.py`); persistiert auf `doc_status.metadata['allowed_roles']` (provenienz-stabiler, am Verarbeitungs-Zeitpunkt lesbarer Kanal via `status_doc.metadata`). Default `None` → keine Verhaltensänderung. Tests: `tests/pipeline/test_allowed_roles_enqueue.py` (7 grün). Noch keine Durchsetzung.
  - Hinweis: `full_docs`/`LIGHTRAG_DOC_FULL` führt `roles` **nicht** mit (PG-`upsert_doc_full` schreibt `meta` ohnehin nicht); `doc_status.metadata` ist der zuverlässige Kanal für S2.
  - Dev-Umgebung: `.venv-acl` (core-Deps + pytest, ohne `[api]`-Extra) im Repo-Root angelegt — nicht committen.
- **S2 — Chunk-Roles stempeln: ✅ ERLEDIGT.** `build_chunks_dict_from_chunking_result(..., allowed_roles=)` stempelt jeden Chunk mit einer Kopie der Doc-Roles; `_process_single_document` liest sie aus `status_doc.metadata['allowed_roles']`. `chunks_vdb.meta_fields` um `roles` erweitert (auch Nicht-PG-Backends). Postgres: `roles TEXT[] NULL` auf `LIGHTRAG_DOC_CHUNKS` + `LIGHTRAG_VDB_CHUNKS`; Migration `_migrate_chunks_add_roles` (additiv, idempotent, GIN-Index auf VDB-Tabelle); Upsert-SQL (`upsert_text_chunk`, `upsert_chunk`) + Builder-Tupel um `roles` ergänzt; `get_by_id(s)_text_chunks` liefern `roles`; Helper `_normalize_roles_for_storage`. Tests: `tests/pipeline/test_allowed_roles_chunk_stamping.py` (4 grün, backend-agnostisch); SQL-Spalten/Platzhalter-Alignment verifiziert. PG-SQL gegen echte DB erst in S9.
- **S3 — Record-Roles: ✅ ERLEDIGT.** `extract_entities._process_single_content` stempelt vor dem Return die Chunk-`roles` auf jeden Entity-/Relation-Record (inkl. multimodal injizierter). Kein Schema, rein `operate.py`.
- **S4 — Merge Provenienz + Fragmente: ✅ ERLEDIGT (mit Architektur-Abweichung, s. u.).** `_merge_nodes_then_upsert`/`_merge_edges_then_upsert`: pro Quelle ein Beschreibungs-Fragment `{source_id, description, roles}`; Roles als **open-dominante Union** (`union_roles_from_fragments`); Fragmente + Roles als Node/Edge-Properties `desc_fragments`/`roles` (Graph) und Entity/Relation-VDB-`roles`. Legacy-Nodes (Pre-RBAC-Blob, keine Fragmente) → ein **offenes** Fragment (bleiben sichtbar). Reine Helfer in neuem Modul `lightrag/acl.py`. Summary-Blob bleibt (Anzeige/Embedding). PG: `roles TEXT[]` auf `LIGHTRAG_VDB_ENTITY`/`LIGHTRAG_VDB_RELATION` + GIN-Index (Migration erweitert); `upsert_entity`/`upsert_relationship`-SQL + Builder um `roles`.
- **S5 — Vektor-Pre-Filter (DB): ✅ ERLEDIGT.** `PGVectorStorage.query(..., user_roles=None)` + `supports_acl_prefilter=True`; `QueryParam.user_roles`; `chunks`/`entities`/`relationships`-SQL-Templates um `($5::text[] IS NULL OR roles IS NULL OR roles && $5::text[])`; Param-Binding via `_normalize_roles_for_storage`. Andere Backends behalten ihre Signatur (kwarg nur an PG).
- **S6 — Graph-Traversierung: ✅ ERLEDIGT.** Backend-agnostischer Post-Filter `_acl_filter_graph_elements` (verwirft Nodes/Edges ohne Rollen-Overlap, baut Description aus Fragmenten neu) in `_get_node_data`, `_find_most_related_edges_from_entities`, `_get_edge_data`, `_find_most_related_entities_from_relationships`. Filterung VOR Edge-/Entity-Expansion, damit verbotene Elemente nicht weiter traversieren.
- **S7 — Kontext-/Beschreibungs-Assembly + Guard: ✅ ERLEDIGT.** Chunk-Filter `_acl_filter_chunks` in `_get_vector_context` + `_find_related_text_unit_from_*`. Description-Assembly aus erlaubten Fragmenten passiert bereits in `_acl_filter_graph_elements` (nie der gemischte Blob, sobald `user_roles` gesetzt). Hard-Reject-Guard `LightRAG._enforce_acl_capability` in `aquery_llm`/`aquery_data`; `aquery_data` reicht `user_roles` jetzt in die Retrieval-Kopie. VDB-Pre-Filter-Call via `_acl_vdb_query` (kwarg nur an Pre-Filter-fähige Backends).
- **S8 — REST-API: ✅ ERLEDIGT.** `allowed_roles` in `InsertTextRequest`/`InsertTextsRequest` → `pipeline_index_texts` → `apipeline_enqueue_documents`. `user_roles` in `QueryRequest` → `to_query_params` → `QueryParam` (automatisch). Verifiziert per `py_compile` (kein `fastapi` im Dev-venv).
- **S9 — E2E-Leak-Test grün + Doku: ✅ ERLEDIGT.** `tests/pipeline/test_acl_leak_e2e.py` (grün, Default-Backends), `tests/pipeline/test_acl_helpers.py` (9 grün). Doku/Leak-Risiken hier aktualisiert.

### 4.1 Bewusste Architektur-Abweichung vom ursprünglichen Plan

1. **Beschreibungs-Fragmente als Node/Edge-Property statt separater `LIGHTRAG_DESC_FRAGMENTS`-Tabelle.** Nodes/Edges tragen `desc_fragments` (JSON `[{source_id, description, roles}]`) + `roles` als Properties. Begründung: Beide Graph-Backends (NetworkX **und** AGE/Postgres) führen beliebige Properties verlustfrei; damit ist die Anti-Leak-Assembly backend-agnostisch und der E2E-Leak-Test **ohne Live-Postgres** ausführbar. Provenienz-pro-Quelle-Semantik ist identisch erfüllt.
2. **Durchsetzung backend-agnostisch (Post-Filter) + PG-Pre-Filter als Defense-in-Depth.** Die Korrektheits-Garantie (kein Leak) liegt im Post-Retrieval-Filter in `operate.py`, der auf den round-trippenden `roles`-Metadaten/Properties arbeitet und auf **allen** Backends greift. Der PG-`&&`-Pre-Filter entfernt verbotene Zeilen zusätzlich schon im Top-k (Performance + mildert Ranking-Inferenz-Leak, Risiko #1). **NanoVectorDB/NetworkX wurden nicht verändert** — sie liefern die Roles ohnehin verlustfrei zurück.
3. **„Hart ablehnen" als Allowlist umgesetzt.** `_enforce_acl_capability` erlaubt nur verifizierte Backends (PG, NanoVectorDB, NetworkX, JSON-KV/PG-KV); jedes andere Backend mit gesetztem `user_roles` wirft einen Fehler statt still ungefiltert zu liefern. Damit ist der Geist der Entscheidung („nie still ungefiltert") gewahrt und die Default-Backends sind testbar abgesichert.

---

## 5. Verifikation — Leak-Szenario (E2E-Test)

Zwei Dokumente nennen **dieselbe** Entität „ACME“:
- Doc A `allowed_roles=['public']`: „ACME ist ein Hersteller.“ + Relation `ACME—public_fact`.
- Doc B `allowed_roles=['confidential']`: „ACME plant Übernahme X.“ + Relation `ACME—secret_deal`.

`public`-User-Query prüft:
- (a) **sieht** Entität „ACME“ (hat erlaubte Quelle Doc A),
- (b) **kein** vertrauliches Fragment („Übernahme X“) in der Beschreibung,
- (c) **keine** vertrauliche Relation `secret_deal` im Kontext,
- (d) **kein** vertraulicher Chunk in `naive`/`mix`-Kontext.
`confidential`-User sieht alles. Query **ohne** `user_roles` sieht alles (Kompatibilität).

---

## 6. Verbleibende Leak-Risiken & offene Design-Entscheidungen

> **Status nach Umsetzung:** Der Haupt-Leak (gemischter Beschreibungs-Blob über
> Rollengrenzen) ist geschlossen — Beschreibungen werden bei gesetztem
> `user_roles` ausschließlich aus erlaubten Fragmenten neu gebaut, verbotene
> Nodes/Edges/Chunks werden verworfen. Die folgenden Rest-Risiken bleiben
> bestehen und sind bewusst dokumentiert:

1. **Embedding-Inferenz:** `LIGHTRAG_VDB_ENTITY.content` embedded den Summary-Blob (inkl. vertraulichem Text). Rückgabe ist nur `entity_name`, aber Treffer-Ranking kann durch vertraulichen Text beeinflusst sein (Existenz-/Themen-Inferenz). Pre-Filter auf `roles` verhindert Treffer rein-vertraulicher Entitäten; bei gemischten Entitäten bleibt Restinferenz. **Option:** separate, rollen-spezifische Embeddings je Fragment (teurer).
2. **Summary-Blob als Restquelle:** Node-`description` bleibt gemischt gespeichert. Muss garantiert **nie** ungefiltert in den Kontext (S7). Risiko bei Code-Pfaden, die `get_node().description` direkt nutzen (z. B. Graph-Visualizer-API, `get_knowledge_graph`) — diese separat absichern oder dokumentieren.
3. **LLM-Cache:** `LIGHTRAG_LLM_CACHE` kann vertrauliche Extraktions-/Summary-Ausgaben enthalten; Query-Cache-Keys müssen `user_roles` einschließen, sonst Cross-Rollen-Cache-Leak.
4. **Keywords/Rerank:** High-/Low-Level-Keywords und Rerank-Eingaben dürfen keine vertraulichen Fragmente enthalten.
5. **Rollen-Hierarchie:** Modell ist flach (OR). Keine Vererbung/Negation/Deny-Regeln — bewusst offen gelassen.
6. **Truncation-Wechselwirkung:** `max_source_ids_per_entity` (FIFO/KEEP) kann Quellen kappen — die zugehörigen Roles dürfen dabei **nicht** verloren gehen (Union getrennt von der gekappten `source_id`-Anzeige führen).
7. **Andere Backends:** Durchsetzung erfolgt backend-agnostisch (Post-Filter) auf den verifizierten Defaults (PG, NanoVectorDB, NetworkX, JSON-KV); PG zusätzlich per DB-Pre-Filter. Unverifizierte Backends (Neo4j, Milvus, Qdrant, Mongo, Redis …) mit gesetztem `user_roles` werden **hart abgelehnt** (`_enforce_acl_capability`-Allowlist). Kein stilles ungefiltertes Ausliefern. Wer ein weiteres Backend freischalten will, muss verifizieren, dass `roles` verlustfrei round-trippt, und es zur Allowlist hinzufügen (oder `supports_acl_prefilter` setzen).
9. **`pick_by_vector_similarity` (KG-Chunk-Auswahl):** ruft `chunks_vdb.query` ohne `user_roles` (kein PG-Pre-Filter an dieser Stelle). Korrektheit bleibt gewahrt, weil die ausgewählten Chunks danach in `_find_related_text_unit_from_*` per `_acl_filter_chunks` gefiltert werden; auf PG entgeht hier lediglich die Pre-Filter-Optimierung. Optional nachrüstbar.
10. **Graph-Visualizer / `get_knowledge_graph`:** liefert Node-`description` (gemischter Blob) und neue Properties `roles`/`desc_fragments` direkt aus dem Graph — **ohne** `user_roles`-Filter. Diese Endpunkte sind nicht ACL-gefiltert und müssen separat abgesichert oder als Admin-only dokumentiert werden.
8. **Deletion/Update:** Beim Doc-Delete/Re-Ingest müssen Fragmente + Roles-Union konsistent zurückgerechnet werden (Union ist nicht trivial dekrementierbar) — Strategie: bei Update Roles aus verbleibenden Fragmenten neu aggregieren.
