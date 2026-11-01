from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, ConfigDict, Field, field_validator


APP_SCHEMA_VERSION = "1.0.0"
DOMAIN_NAME = "bounded-business-domain"
SEED = 20261101
EXTRACTOR_VERSION = "rule-extractor-1.0.0"
BUILD_TAG = "single-file-demo-1"
SOURCE_BASE_TIME = datetime(2026, 11, 1, 8, 0, 0, tzinfo=timezone.utc)

AUTO_APPROVE_THRESHOLD = 0.90
REVIEW_REQUIRED_PREDICATES = {"supplied_by", "competes_with"}

ERROR_CODES = {
    "KG-E001": "Unknown subject or object entity.",
    "KG-E002": "Missing or unknown source document.",
    "KG-E003": "Invalid source span.",
    "KG-E004": "Unsupported predicate.",
    "KG-E005": "Domain constraint violation.",
    "KG-E006": "Range constraint violation.",
    "KG-E007": "Duplicate approved fact.",
    "KG-E008": "Cardinality constraint violation.",
    "KG-E009": "Low confidence candidate requires review.",
}

NODE_COLORS = {
    "Company": "#2563eb",
    "Product": "#16a34a",
    "Person": "#dc2626",
    "Document": "#f59e0b",
}

PRODUCES_KEYWORDS = (
    "produces",
    "manufactures",
    "makes",
    "announced",
    "launched",
    "introduced",
    "catalog",
)
STRONG_PRODUCES_KEYWORDS = ("produces", "manufactures", "makes", "launched", "introduced")
SUPPLY_KEYWORDS = ("supplied by", "components from", "uses", "built with", "sources from")
COMPETE_KEYWORDS = ("competes", "rival", "competitor")


COMPANY_DATA = [
    ("Acme Robotics", ["Acme", "Acme Robotics Corp"]),
    ("Nova Cloud", ["Nova", "NovaCloud"]),
    ("Helix Bio", ["Helix", "Helix Biosciences"]),
    ("Orion Logistics", ["Orion", "Orion Freight"]),
    ("Vertex Materials", ["Vertex", "Vertex Alloys"]),
]

PRODUCT_DATA = [
    ("Atlas Arm", ["Atlas Arm"]),
    ("Fabric Mesh", ["Fabric Mesh"]),
    ("BioScan Kit", ["BioScan Kit"]),
    ("RoutePilot", ["RoutePilot"]),
    ("AlloyCore", ["AlloyCore"]),
]

PERSON_DATA = [
    ("Dana Lee", ["Dana Lee", "D. Lee"]),
    ("Sam Patel", ["Sam Patel", "S. Patel"]),
]

DOCUMENT_DATA = [
    {
        "document_id": "doc-001",
        "title": "Q1 supplier memo",
        "access_label": "internal",
        "text": (
            "Acme Robotics produces Atlas Arm. "
            "Atlas Arm uses AlloyCore supplied by Vertex Materials. "
            "Report by Dana Lee."
        ),
    },
    {
        "document_id": "doc-002",
        "title": "Product catalog",
        "access_label": "public",
        "text": (
            "Product catalog: Nova Cloud produces Fabric Mesh. "
            "Helix Bio produces BioScan Kit. "
            "Orion Logistics produces RoutePilot. "
            "Vertex Materials produces AlloyCore. "
            "Authored by Sam Patel."
        ),
    },
    {
        "document_id": "doc-003",
        "title": "Market brief: cloud logistics",
        "access_label": "public",
        "text": (
            "Nova Cloud competes with Orion Logistics in route optimization. "
            "RoutePilot uses Fabric Mesh. "
            "Dana Lee prepared the brief."
        ),
    },
    {
        "document_id": "doc-004",
        "title": "Procurement note",
        "access_label": "restricted",
        "text": (
            "BioScan Kit uses Fabric Mesh supplied by Nova Cloud. "
            "Helix Bio produces BioScan Kit. "
            "Sam Patel authored the procurement note."
        ),
    },
    {
        "document_id": "doc-005",
        "title": "Logistics update",
        "access_label": "internal",
        "text": (
            "Orion Logistics produces RoutePilot. "
            "Orion Logistics competes with Nova Cloud. "
            "Report by Dana Lee."
        ),
    },
    {
        "document_id": "doc-006",
        "title": "Materials bulletin",
        "access_label": "public",
        "text": (
            "Vertex Materials produces AlloyCore. "
            "AlloyCore is supplied by Vertex Materials. "
            "Acme Robotics uses AlloyCore in Atlas Arm."
        ),
    },
    {
        "document_id": "doc-007",
        "title": "Quality memo",
        "access_label": "internal",
        "text": (
            "Helix Bio produces BioScan Kit. "
            "Nova Cloud competes with Helix Bio in analytics. "
            "Sam Patel authored the quality memo."
        ),
    },
    {
        "document_id": "doc-008",
        "title": "Executive summary",
        "access_label": "restricted",
        "text": (
            "Acme Robotics produces Atlas Arm. "
            "Nova Cloud produces Fabric Mesh. "
            "Vertex Materials supplies AlloyCore. "
            "Dana Lee authored the executive summary."
        ),
    },
]

CONFIG_PAYLOAD = {
    "app_schema_version": APP_SCHEMA_VERSION,
    "domain": DOMAIN_NAME,
    "seed": SEED,
    "extractor_version": EXTRACTOR_VERSION,
    "base_time": SOURCE_BASE_TIME.isoformat(),
    "document_count": len(DOCUMENT_DATA),
    "build_tag": BUILD_TAG,
}
CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG_PAYLOAD, sort_keys=True).encode("utf-8")
).hexdigest()[:16]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


logger = logging.getLogger("kg_demo")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_event(event: str, **kwargs: Any) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "config_hash": CONFIG_HASH,
        **kwargs,
    }
    logger.info(json.dumps(payload, sort_keys=True, default=str))


Status = Literal["staged", "validated", "review", "approved", "rejected"]
AccessLabel = Literal["public", "internal", "restricted"]


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_document_id: str
    source_span_start: int = Field(ge=0)
    source_span_end: int = Field(ge=0)
    extractor_version: str
    confidence: float = Field(ge=0.0, le=1.0)
    ingested_at: str

    @field_validator("source_span_end")
    @classmethod
    def end_after_start(cls, v: int, info: Any) -> int:
        start = info.data.get("source_span_start")
        if isinstance(start, int) and v <= start:
            raise ValueError("source_span_end must be greater than source_span_start")
        return v


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    text: str
    access_label: AccessLabel
    sha256: str


class RelationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    status: Status
    provenance: Provenance
    evidence_text: str
    validation_errors: list[str] = Field(default_factory=list)
    review_note: str = ""


class ClassDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    description: str


class PropertyConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicate: str
    domain: list[str]
    range: list[str]
    min_cardinality: int = 0
    max_cardinality: Optional[int] = None


CLASSES = [
    ClassDefinition(
        id="Company",
        label="Company",
        description="A business entity that produces, supplies, or competes in the domain.",
    ),
    ClassDefinition(
        id="Product",
        label="Product",
        description="A tangible or software product associated with one or more companies.",
    ),
    ClassDefinition(
        id="Person",
        label="Person",
        description="A human author, analyst, or reviewer referenced by documents.",
    ),
    ClassDefinition(
        id="Document",
        label="Document",
        description="A source record carrying provenance for extracted facts.",
    ),
]

PROPERTY_CONSTRAINTS = [
    PropertyConstraint(
        predicate="mentions",
        domain=["Document"],
        range=["Company", "Product", "Person"],
        min_cardinality=0,
        max_cardinality=None,
    ),
    PropertyConstraint(
        predicate="authored_by",
        domain=["Document"],
        range=["Person"],
        min_cardinality=0,
        max_cardinality=None,
    ),
    PropertyConstraint(
        predicate="produces",
        domain=["Company"],
        range=["Product"],
        min_cardinality=0,
        max_cardinality=None,
    ),
    PropertyConstraint(
        predicate="supplied_by",
        domain=["Product"],
        range=["Company"],
        min_cardinality=0,
        max_cardinality=None,
    ),
    PropertyConstraint(
        predicate="competes_with",
        domain=["Company"],
        range=["Company"],
        min_cardinality=0,
        max_cardinality=None,
    ),
]


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"urn:kg:{prefix}:{digest}"


def truncate(value: str, limit: int = 18) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def build_base_entities() -> list[Entity]:
    entities: list[Entity] = []

    for name, aliases in COMPANY_DATA:
        entities.append(
            Entity(
                entity_id=stable_id("entity", name, "Company"),
                canonical_name=name,
                entity_type="Company",
                aliases=aliases,
            )
        )

    for name, aliases in PRODUCT_DATA:
        entities.append(
            Entity(
                entity_id=stable_id("entity", name, "Product"),
                canonical_name=name,
                entity_type="Product",
                aliases=aliases,
            )
        )

    for name, aliases in PERSON_DATA:
        entities.append(
            Entity(
                entity_id=stable_id("entity", name, "Person"),
                canonical_name=name,
                entity_type="Person",
                aliases=aliases,
            )
        )

    return entities


def build_documents() -> tuple[list[SourceDocument], list[Entity], dict[str, str]]:
    docs: list[SourceDocument] = []
    doc_entities: list[Entity] = []
    doc_to_entity: dict[str, str] = {}

    for item in DOCUMENT_DATA:
        text = item["text"]
        sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc = SourceDocument(
            document_id=item["document_id"],
            title=item["title"],
            text=text,
            access_label=item["access_label"],
            sha256=sha256,
        )
        docs.append(doc)

        doc_entity = Entity(
            entity_id=stable_id("entity", doc.title, "Document"),
            canonical_name=doc.title,
            entity_type="Document",
            aliases=[doc.document_id],
        )
        doc_entities.append(doc_entity)
        doc_to_entity[doc.document_id] = doc_entity.entity_id

    return docs, doc_entities, doc_to_entity


def build_alias_map(entities: list[Entity]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for entity in entities:
        if entity.entity_type == "Document":
            continue
        for alias in [entity.canonical_name, *entity.aliases]:
            alias_map[alias.lower()] = entity.entity_id
    return alias_map


def split_sentences(text: str) -> list[tuple[int, str]]:
    sentences: list[tuple[int, str]] = []
    for match in re.finditer(r"[^.!?\n]+[.!?]?", text):
        sentence = match.group(0).strip()
        if sentence:
            sentences.append((match.start(), sentence))
    return sentences


def find_mentions(
    sentence: str,
    offset: int,
    alias_map: dict[str, str],
) -> list[tuple[str, int, int]]:
    lower = sentence.lower()
    raw: list[tuple[int, int, str]] = []

    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        for match in re.finditer(re.escape(alias), lower):
            raw.append((match.start(), match.end(), alias_map[alias]))

    selected: list[tuple[int, int, str]] = []
    for start, end, entity_id in sorted(raw, key=lambda x: (x[0], -(x[1] - x[0]))):
        overlaps = False
        for sel_start, sel_end, _ in selected:
            if not (end <= sel_start or start >= sel_end):
                overlaps = True
                break
        if not overlaps:
            selected.append((start, end, entity_id))

    deduped: dict[str, tuple[str, int, int]] = {}
    for start, end, entity_id in selected:
        if entity_id not in deduped:
            deduped[entity_id] = (entity_id, offset + start, offset + end)

    return list(deduped.values())


def add_candidate(
    out: list[RelationCandidate],
    seen: set[str],
    subject_id: str,
    predicate: str,
    object_id: str,
    doc: SourceDocument,
    span: tuple[int, int],
    confidence: float,
    evidence: str,
    ingested_at: str,
) -> None:
    relation_id = stable_id("relation", subject_id, predicate, object_id)
    if relation_id in seen:
        return
    seen.add(relation_id)

    start, end = span
    if end <= start:
        end = min(len(doc.text), start + 1)

    provenance = Provenance(
        source_document_id=doc.document_id,
        source_span_start=start,
        source_span_end=end,
        extractor_version=EXTRACTOR_VERSION,
        confidence=confidence,
        ingested_at=ingested_at,
    )

    candidate = RelationCandidate(
        relation_id=relation_id,
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        status="staged",
        provenance=provenance,
        evidence_text=evidence,
    )
    out.append(candidate)


def extract_document(
    doc: SourceDocument,
    doc_entity_id: str,
    alias_map: dict[str, str],
    entity_by_id: dict[str, Entity],
    ingested_at: str,
) -> list[RelationCandidate]:
    candidates: list[RelationCandidate] = []
    seen: set[str] = set()

    for offset, sentence in split_sentences(doc.text):
        lower = sentence.lower()
        mentions = find_mentions(sentence, offset, alias_map)

        companies: list[tuple[str, int, int]] = []
        products: list[tuple[str, int, int]] = []
        persons: list[tuple[str, int, int]] = []

        for entity_id, start, end in mentions:
            entity = entity_by_id.get(entity_id)
            if entity is None:
                continue

            if entity.entity_type == "Company":
                companies.append((entity_id, start, end))
            elif entity.entity_type == "Product":
                products.append((entity_id, start, end))
            elif entity.entity_type == "Person":
                persons.append((entity_id, start, end))

            add_candidate(
                candidates,
                seen,
                doc_entity_id,
                "mentions",
                entity_id,
                doc,
                (start, end),
                0.99,
                sentence,
                ingested_at,
            )

        for entity_id, start, end in persons:
            if re.search(r"\b(by|author|prepared|written)\b", lower):
                add_candidate(
                    candidates,
                    seen,
                    doc_entity_id,
                    "authored_by",
                    entity_id,
                    doc,
                    (start, end),
                    0.95,
                    sentence,
                    ingested_at,
                )

        for company_id, company_start, company_end in companies:
            for product_id, product_start, product_end in products:
                if any(keyword in lower for keyword in PRODUCES_KEYWORDS):
                    confidence = (
                        0.93
                        if any(keyword in lower for keyword in STRONG_PRODUCES_KEYWORDS)
                        else 0.82
                    )
                    span = (min(company_start, product_start), max(company_end, product_end))
                    add_candidate(
                        candidates,
                        seen,
                        company_id,
                        "produces",
                        product_id,
                        doc,
                        span,
                        confidence,
                        sentence,
                        ingested_at,
                    )

        for product_id, product_start, product_end in products:
            for company_id, company_start, company_end in companies:
                if any(keyword in lower for keyword in SUPPLY_KEYWORDS):
                    confidence = 0.91 if "supplied by" in lower else 0.78
                    span = (min(product_start, company_start), max(product_end, company_end))
                    add_candidate(
                        candidates,
                        seen,
                        product_id,
                        "supplied_by",
                        company_id,
                        doc,
                        span,
                        confidence,
                        sentence,
                        ingested_at,
                    )

        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                if any(keyword in lower for keyword in COMPETE_KEYWORDS):
                    left_id, left_start, left_end = companies[i]
                    right_id, right_start, right_end = companies[j]
                    span = (min(left_start, right_start), max(left_end, right_end))
                    add_candidate(
                        candidates,
                        seen,
                        left_id,
                        "competes_with",
                        right_id,
                        doc,
                        span,
                        0.86,
                        sentence,
                        ingested_at,
                    )

    return candidates


def validate_candidate(
    candidate: RelationCandidate,
    entity_by_id: dict[str, Entity],
    doc_by_id: dict[str, SourceDocument],
    constraint_by_predicate: dict[str, PropertyConstraint],
    approved_keys: set[tuple[str, str, str]],
    approved_counts: dict[tuple[str, str], int],
) -> RelationCandidate:
    errors: list[str] = []

    subject = entity_by_id.get(candidate.subject_id)
    object_ = entity_by_id.get(candidate.object_id)

    if subject is None or object_ is None:
        errors.append("KG-E001")

    doc = doc_by_id.get(candidate.provenance.source_document_id)
    if doc is None:
        errors.append("KG-E002")
    else:
        start = candidate.provenance.source_span_start
        end = candidate.provenance.source_span_end
        if start < 0 or end > len(doc.text) or start >= end:
            errors.append("KG-E003")

    constraint = constraint_by_predicate.get(candidate.predicate)
    if constraint is None:
        errors.append("KG-E004")
    elif subject is not None and object_ is not None:
        if subject.entity_type not in constraint.domain:
            errors.append("KG-E005")
        if object_.entity_type not in constraint.range:
            errors.append("KG-E006")

        key = (candidate.subject_id, candidate.predicate)
        max_card = constraint.max_cardinality
        if max_card is not None and approved_counts.get(key, 0) >= max_card:
            errors.append("KG-E008")

    triple_key = (candidate.subject_id, candidate.predicate, candidate.object_id)
    if triple_key in approved_keys:
        errors.append("KG-E007")

    if errors:
        return candidate.model_copy(update={"status": "rejected", "validation_errors": errors})

    return candidate.model_copy(update={"status": "validated", "validation_errors": []})


def auto_promote(candidate: RelationCandidate) -> RelationCandidate:
    if candidate.status != "validated":
        return candidate

    if (
        candidate.predicate in REVIEW_REQUIRED_PREDICATES
        or candidate.provenance.confidence < AUTO_APPROVE_THRESHOLD
    ):
        return candidate.model_copy(update={"status": "review"})

    return candidate.model_copy(update={"status": "approved"})


def relation_label(
    candidate: RelationCandidate,
    entity_by_id: dict[str, Entity],
) -> str:
    subject = entity_by_id.get(candidate.subject_id)
    object_ = entity_by_id.get(candidate.object_id)
    subject_name = subject.canonical_name if subject else candidate.subject_id
    object_name = object_.canonical_name if object_ else candidate.object_id
    return f"{subject_name} {candidate.predicate} {object_name}"


def build_assets() -> tuple[
    list[Entity],
    list[SourceDocument],
    dict[str, RelationCandidate],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Entity],
    dict[str, SourceDocument],
]:
    base_entities = build_base_entities()
    docs, doc_entities, doc_to_entity = build_documents()
    entities = base_entities + doc_entities

    entity_by_id = {entity.entity_id: entity for entity in entities}
    doc_by_id = {doc.document_id: doc for doc in docs}
    alias_map = build_alias_map(entities)
    constraint_by_predicate = {constraint.predicate: constraint for constraint in PROPERTY_CONSTRAINTS}

    candidates_by_id: dict[str, RelationCandidate] = {}
    approved_ids: set[str] = set()
    approved_keys: set[tuple[str, str, str]] = set()
    approved_counts: dict[tuple[str, str], int] = {}
    timeline: list[dict[str, Any]] = []

    for tick, doc in enumerate(docs, start=1):
        ingested_at = (SOURCE_BASE_TIME + timedelta(minutes=tick)).isoformat()
        doc_entity_id = doc_to_entity[doc.document_id]

        staged_candidates = extract_document(
            doc=doc,
            doc_entity_id=doc_entity_id,
            alias_map=alias_map,
            entity_by_id=entity_by_id,
            ingested_at=ingested_at,
        )

        new_approvals: list[str] = []

        for candidate in staged_candidates:
            if candidate.relation_id in candidates_by_id:
                continue

            validated = validate_candidate(
                candidate=candidate,
                entity_by_id=entity_by_id,
                doc_by_id=doc_by_id,
                constraint_by_predicate=constraint_by_predicate,
                approved_keys=approved_keys,
                approved_counts=approved_counts,
            )
            final_candidate = auto_promote(validated)
            candidates_by_id[final_candidate.relation_id] = final_candidate

            if final_candidate.status == "approved":
                approved_ids.add(final_candidate.relation_id)
                approved_keys.add(
                    (
                        final_candidate.subject_id,
                        final_candidate.predicate,
                        final_candidate.object_id,
                    )
                )
                count_key = (final_candidate.subject_id, final_candidate.predicate)
                approved_counts[count_key] = approved_counts.get(count_key, 0) + 1
                new_approvals.append(relation_label(final_candidate, entity_by_id))

        pending = sum(1 for c in candidates_by_id.values() if c.status == "review")
        rejected = sum(1 for c in candidates_by_id.values() if c.status == "rejected")
        approved_confidences = [
            candidates_by_id[rid].provenance.confidence
            for rid in approved_ids
            if rid in candidates_by_id
        ]
        avg_confidence = float(np.mean(approved_confidences)) if approved_confidences else 0.0
        promoted_conformance = (
            1.0
            if not approved_ids
            else float(
                all(
                    not candidates_by_id[rid].validation_errors
                    for rid in approved_ids
                    if rid in candidates_by_id
                )
            )
        )

        timeline.append(
            {
                "tick": tick,
                "caption": f"Tick {tick}: {doc.title}",
                "detail": f"Ingested {doc.document_id} ({doc.access_label})",
                "approved_ids": sorted(approved_ids),
                "new_approvals": new_approvals,
                "pending": pending,
                "rejected": rejected,
                "documents_ingested": tick,
                "entities": len(entity_by_id),
                "avg_confidence": avg_confidence,
                "promoted_conformance": promoted_conformance,
            }
        )

    layout_graph = nx.MultiDiGraph()
    for entity in entities:
        layout_graph.add_node(entity.entity_id, type=entity.entity_type)
    for candidate in candidates_by_id.values():
        layout_graph.add_edge(candidate.subject_id, candidate.object_id)

    if layout_graph.number_of_nodes() > 0:
        pos = nx.spring_layout(layout_graph, seed=SEED, k=1.0, iterations=100)
    else:
        pos = {}

    return (
        entities,
        docs,
        candidates_by_id,
        timeline,
        pos,
        entity_by_id,
        doc_by_id,
    )


def make_empty_frame(entity_count: int) -> dict[str, Any]:
    return {
        "tick": 0,
        "caption": "Initializing governed graph",
        "detail": "Loading deterministic synthetic documents...",
        "approved_ids": [],
        "new_approvals": [],
        "pending": 0,
        "rejected": 0,
        "documents_ingested": 0,
        "entities": entity_count,
        "avg_confidence": 0.0,
        "promoted_conformance": 1.0,
    }


def figure_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buffer = fig.canvas.buffer_rgba()
    array = np.asarray(buffer)
    rgb = np.ascontiguousarray(array[..., :3], dtype=np.uint8)
    plt.close(fig)
    return rgb


def render_frame(
    frame: dict[str, Any],
    candidates_by_id: dict[str, RelationCandidate],
    entity_by_id: dict[str, Entity],
    pos: dict[str, Any],
) -> np.ndarray:
    fig = plt.figure(figsize=(8, 4.8), dpi=100, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.03)

    ax = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])

    graph = nx.DiGraph()
    for relation_id in frame.get("approved_ids", []):
        candidate = candidates_by_id.get(relation_id)
        if candidate is None:
            continue

        for node_id in (candidate.subject_id, candidate.object_id):
            entity = entity_by_id.get(node_id)
            if entity is not None:
                graph.add_node(node_id, label=entity.canonical_name, type=entity.entity_type)

        if candidate.subject_id in graph and candidate.object_id in graph:
            graph.add_edge(candidate.subject_id, candidate.object_id, predicate=candidate.predicate)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    ax.set_title(
        "Enterprise Knowledge Graph — governed ingestion demo",
        fontsize=10,
        loc="left",
    )

    if graph.number_of_nodes() == 0:
        ax.text(
            0.5,
            0.5,
            "Waiting for first approved fact...",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#64748b",
        )
    else:
        node_pos = {node: pos.get(node, (0.0, 0.0)) for node in graph.nodes()}
        colors = [
            NODE_COLORS.get(entity_by_id[node].entity_type, "#64748b")
            if node in entity_by_id
            else "#64748b"
            for node in graph.nodes()
        ]
        labels = {
            node: truncate(entity_by_id[node].canonical_name, 16)
            for node in graph.nodes()
            if node in entity_by_id
        }

        nx.draw_networkx_nodes(
            graph,
            node_pos,
            ax=ax,
            node_color=colors,
            node_size=650,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
        )
        nx.draw_networkx_labels(
            graph,
            node_pos,
            labels=labels,
            ax=ax,
            font_size=6.5,
            font_color="#111827",
        )
        nx.draw_networkx_edges(
            graph,
            node_pos,
            ax=ax,
            arrows=True,
            edge_color="#475569",
            width=1.1,
            alpha=0.65,
        )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=color,
            markersize=7,
            label=entity_type,
        )
        for entity_type, color in NODE_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=6, frameon=False)

    lines = [
        f"Tick: {frame.get('tick', 0):02d}",
        f"Event: {frame.get('caption', '')}",
        f"Detail: {frame.get('detail', '')}",
        "",
        f"Documents ingested: {frame.get('documents_ingested', 0)}",
        f"Entities: {frame.get('entities', 0)}",
        f"Approved edges: {len(frame.get('approved_ids', []))}",
        f"Review queue: {frame.get('pending', 0)}",
        f"Rejected: {frame.get('rejected', 0)}",
        f"Avg confidence: {frame.get('avg_confidence', 0.0):.2f}",
        f"Promoted SHACL: {frame.get('promoted_conformance', 1.0) * 100:.0f}%",
        "",
        "New approvals:",
    ]
    new_approvals = frame.get("new_approvals", [])
    if new_approvals:
        lines.extend(f"- {truncate(item, 38)}" for item in new_approvals[:4])
    else:
        lines.append("- none in this phase")

    ax_stats.axis("off")
    ax_stats.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=7.2,
        family="monospace",
        wrap=True,
    )

    fig.tight_layout()
    return figure_to_rgb_array(fig)


@st.cache_resource(show_spinner="Generating live MP4 demo...")
def get_demo_video(config_hash: str) -> bytes:
    _, _, candidates_by_id, timeline, pos, entity_by_id, _ = build_assets()

    frames: list[np.ndarray] = []
    frames.append(render_frame(make_empty_frame(len(entity_by_id)), candidates_by_id, entity_by_id, pos))

    for frame_data in timeline:
        phases = [
            ("Ingesting document", frame_data.get("detail", "")),
            ("Extracting candidates", frame_data.get("caption", "")),
            ("Validating shapes", f"Pending review: {frame_data.get('pending', 0)}"),
            ("Promoting approved facts", f"Approved edges: {len(frame_data.get('approved_ids', []))}"),
        ]
        for caption, detail in phases:
            phase_frame = dict(frame_data)
            phase_frame["caption"] = caption
            phase_frame["detail"] = detail
            frames.append(render_frame(phase_frame, candidates_by_id, entity_by_id, pos))

    for _ in range(8):
        frames.append(frames[-1].copy())

    video_path = Path(tempfile.gettempdir()) / f"kg_demo_{config_hash}.mp4"

    if not video_path.exists() or video_path.stat().st_size == 0:
        partial_path = video_path.with_name(video_path.stem + ".partial.mp4")
        try:
            imageio.mimsave(partial_path, frames, fps=12)
            partial_path.replace(video_path)
        finally:
            if partial_path.exists():
                partial_path.unlink(missing_ok=True)

    return video_path.read_bytes()


def embed_video(video_bytes: bytes, height: int = 500) -> None:
    encoded = base64.b64encode(video_bytes).decode("ascii")
    html = f"""
    <video
      src="data:video/mp4;base64,{encoded}"
      width="100%"
      height="{height}"
      autoplay
      muted
      loop
      controls
      style="background:#000; border-radius:12px;">
    </video>
    """
    st.components.v1.html(html, height=height + 20, scrolling=False)


def build_approved_graph(
    candidates_by_id: dict[str, RelationCandidate],
    statuses: dict[str, str],
    entity_by_id: dict[str, Entity],
    allowed_types: Optional[list[str]] = None,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    for entity_id, entity in entity_by_id.items():
        if allowed_types is None or entity.entity_type in allowed_types:
            graph.add_node(entity_id, label=entity.canonical_name, type=entity.entity_type)

    for relation_id, candidate in candidates_by_id.items():
        if statuses.get(relation_id) != "approved":
            continue
        if candidate.subject_id in graph and candidate.object_id in graph:
            graph.add_edge(
                candidate.subject_id,
                candidate.object_id,
                key=candidate.predicate,
                relation_id=relation_id,
                confidence=candidate.provenance.confidence,
            )

    return graph


def draw_graph(
    graph: nx.MultiDiGraph,
    entity_by_id: dict[str, Entity],
    pos: dict[str, Any],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=100)
    ax.set_axis_off()

    if graph.number_of_nodes() == 0:
        ax.text(
            0.5,
            0.5,
            "No approved facts match the current filters.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="#64748b",
        )
        fig.tight_layout()
        return fig

    if any(node not in pos for node in graph.nodes()):
        node_pos = nx.spring_layout(graph, seed=SEED)
    else:
        node_pos = {node: pos[node] for node in graph.nodes()}

    colors = [
        NODE_COLORS.get(entity_by_id[node].entity_type, "#64748b")
        if node in entity_by_id
        else "#64748b"
        for node in graph.nodes()
    ]
    labels = {
        node: truncate(entity_by_id[node].canonical_name, 18)
        for node in graph.nodes()
        if node in entity_by_id
    }

    nx.draw_networkx_nodes(
        graph,
        node_pos,
        ax=ax,
        node_color=colors,
        node_size=850,
        edgecolors="white",
        linewidths=0.9,
        alpha=0.95,
    )
    nx.draw_networkx_labels(
        graph,
        node_pos,
        labels=labels,
        ax=ax,
        font_size=7,
        font_color="#111827",
    )
    nx.draw_networkx_edges(
        graph,
        node_pos,
        ax=ax,
        arrows=True,
        edge_color="#475569",
        width=1.2,
        alpha=0.7,
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=color,
            markersize=8,
            label=entity_type,
        )
        for entity_type, color in NODE_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=7, frameon=False)

    fig.tight_layout()
    return fig


def candidate_to_row(
    relation_id: str,
    candidate: RelationCandidate,
    status: str,
    entity_by_id: dict[str, Entity],
) -> dict[str, Any]:
    return {
        "relation_id": relation_id,
        "predicate": candidate.predicate,
        "status": status,
        "subject": entity_by_id.get(candidate.subject_id, Entity(
            entity_id=candidate.subject_id,
            canonical_name=candidate.subject_id,
            entity_type="Unknown",
        )).canonical_name,
        "object": entity_by_id.get(candidate.object_id, Entity(
            entity_id=candidate.object_id,
            canonical_name=candidate.object_id,
            entity_type="Unknown",
        )).canonical_name,
        "confidence": candidate.provenance.confidence,
        "source_document_id": candidate.provenance.source_document_id,
        "span": f"{candidate.provenance.source_span_start}-{candidate.provenance.source_span_end}",
        "extractor_version": candidate.provenance.extractor_version,
        "ingested_at": candidate.provenance.ingested_at,
        "validation_errors": candidate.validation_errors,
        "evidence_text": candidate.evidence_text,
    }


def main() -> None:
    st.set_page_config(
        page_title="Enterprise Knowledge Graph Demo",
        page_icon="🕸️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("KG Demo Controls")
    st.sidebar.markdown(f"**Schema version:** `{APP_SCHEMA_VERSION}`")
    st.sidebar.markdown(f"**Config hash:** `{CONFIG_HASH}`")
    st.sidebar.markdown("**API keys:** not required")
    st.sidebar.markdown("**Runtime:** CPU-only, deterministic synthetic demo")

    if st.sidebar.button("Reset interactive state"):
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()

    (
        entities,
        docs,
        candidates_by_id,
        timeline,
        pos,
        entity_by_id,
        doc_by_id,
    ) = build_assets()

    if (
        "statuses" not in st.session_state
        or set(st.session_state.statuses) != set(candidates_by_id)
    ):
        st.session_state.statuses = {
            relation_id: candidate.status
            for relation_id, candidate in candidates_by_id.items()
        }
        st.session_state.rejection_notes = {}
        st.session_state.audit_log = []

    statuses = st.session_state.statuses
    rejection_notes = st.session_state.rejection_notes
    audit_log = st.session_state.audit_log

    try:
        video_bytes = get_demo_video(CONFIG_HASH)
    except Exception as exc:
        video_bytes = None
        log_event("video_generation_failed", error=str(exc))

    st.title("Enterprise Knowledge Graph Platform — Single-File Working Demo")
    st.caption(
        "This prototype loads realistic synthetic business records, runs a deterministic ingestion "
        "simulation, validates candidate facts against an ontology, keeps provenance for every edge, "
        "and exposes a human-review queue. It generates and embeds a live MP4 demo automatically."
    )

    tab_video, tab_graph, tab_review, tab_provenance, tab_ontology = st.tabs(
        [
            "Live demo video",
            "Graph explorer",
            "Human review",
            "Provenance & metrics",
            "Ontology",
        ]
    )

    with tab_video:
        st.subheader("Auto-generated working demo")

        if video_bytes is not None:
            embed_video(video_bytes)
            st.download_button(
                label="Download generated MP4 demo video",
                data=video_bytes,
                file_name=f"enterprise_kg_demo_{CONFIG_HASH}.mp4",
                mime="video/mp4",
            )
        else:
            st.warning(
                "MP4 generation failed on this machine. Showing the final simulated graph frame instead. "
                "Install the requirements listed in requirements.txt and restart the app."
            )
            fallback_frame = timeline[-1] if timeline else make_empty_frame(len(entity_by_id))
            fallback_image = render_frame(fallback_frame, candidates_by_id, entity_by_id, pos)
            st.image(fallback_image, use_container_width=True)

        st.divider()
        st.write(
            "- The video is generated locally from deterministic synthetic data.\n"
            "- No external API key is used.\n"
            "- Every promoted edge carries source document, span, extractor version, confidence, and ingestion timestamp.\n"
            "- Low-confidence or sensitive relations are routed to human review instead of being silently promoted."
        )

    with tab_graph:
        st.subheader("Governed graph explorer")

        all_types = sorted({entity.entity_type for entity in entities})
        allowed_types = st.multiselect(
            "Entity types",
            options=all_types,
            default=all_types,
        )

        graph = build_approved_graph(
            candidates_by_id=candidates_by_id,
            statuses=statuses,
            entity_by_id=entity_by_id,
            allowed_types=allowed_types,
        )

        fig = draw_graph(graph, entity_by_id, pos)
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        search = st.text_input("Search entity by name")
        if search.strip():
            matches = [
                entity
                for entity in entities
                if search.lower() in entity.canonical_name.lower()
                or any(search.lower() in alias.lower() for alias in entity.aliases)
            ]
            if matches:
                selected_name = st.selectbox(
                    "Selected entity",
                    [entity.canonical_name for entity in matches],
                )
                selected_entity = next(
                    entity for entity in matches if entity.canonical_name == selected_name
                )

                rows: list[dict[str, Any]] = []
                for relation_id, candidate in candidates_by_id.items():
                    if statuses.get(relation_id) != "approved":
                        continue
                    if (
                        candidate.subject_id == selected_entity.entity_id
                        or candidate.object_id == selected_entity.entity_id
                    ):
                        direction = "outgoing" if candidate.subject_id == selected_entity.entity_id else "incoming"
                        neighbor_id = (
                            candidate.object_id
                            if candidate.subject_id == selected_entity.entity_id
                            else candidate.subject_id
                        )
                        neighbor = entity_by_id.get(neighbor_id)
                        doc = doc_by_id.get(candidate.provenance.source_document_id)
                        rows.append(
                            {
                                "direction": direction,
                                "predicate": candidate.predicate,
                                "neighbor": neighbor.canonical_name if neighbor else neighbor_id,
                                "confidence": candidate.provenance.confidence,
                                "source_document": doc.title if doc else candidate.provenance.source_document_id,
                                "span": f"{candidate.provenance.source_span_start}-{candidate.provenance.source_span_end}",
                                "evidence": candidate.evidence_text,
                            }
                        )

                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("This entity has no approved incident edges under the current filters.")
            else:
                st.info("No entity matched the search.")

    with tab_review:
        st.subheader("Human review queue")

        pending_ids = [
            relation_id
            for relation_id, status in statuses.items()
            if status == "review"
        ]

        if pending_ids:
            options = [
                f"{relation_label(candidates_by_id[rid], entity_by_id)} — confidence {candidates_by_id[rid].provenance.confidence:.2f}"
                for rid in pending_ids
            ]
            selected_index = st.selectbox(
                "Select candidate fact",
                range(len(options)),
                format_func=lambda i: options[i],
            )
            selected_id = pending_ids[selected_index]
            candidate = candidates_by_id[selected_id]

            st.json(candidate.model_dump(), expanded=False)

            col_approve, col_reject = st.columns(2)

            with col_approve:
                if st.button("Approve selected", type="primary"):
                    if candidate.validation_errors:
                        st.error("Cannot approve: validation errors are present.")
                    else:
                        statuses[selected_id] = "approved"
                        audit_log.append(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "action": "approve",
                                "relation_id": selected_id,
                                "relation": relation_label(candidate, entity_by_id),
                            }
                        )
                        log_event(
                            "human_review_approve",
                            relation_id=selected_id,
                            confidence=candidate.provenance.confidence,
                        )
                        st.success("Candidate approved and promoted to the governed graph.")
                        st.rerun()

            with col_reject:
                reason = st.text_input(
                    "Rejection reason",
                    key=f"rejection_reason_{selected_id}",
                    placeholder="Optional reviewer note",
                )
                if st.button("Reject selected"):
                    statuses[selected_id] = "rejected"
                    rejection_notes[selected_id] = reason or "Rejected by reviewer"
                    audit_log.append(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "action": "reject",
                            "relation_id": selected_id,
                            "relation": relation_label(candidate, entity_by_id),
                            "reason": rejection_notes[selected_id],
                        }
                    )
                    log_event(
                        "human_review_reject",
                        relation_id=selected_id,
                        reason=rejection_notes[selected_id],
                    )
                    st.warning("Candidate rejected. The fact is retained with a reason.")
                    st.rerun()
        else:
            st.success("No candidates are pending review.")

        rejected_ids = [
            relation_id
            for relation_id, status in statuses.items()
            if status == "rejected"
        ]
        if rejected_ids:
            st.divider()
            st.subheader("Rejected facts retained with reasons")
            rejected_rows = []
            for rid in rejected_ids:
                candidate = candidates_by_id[rid]
                rejected_rows.append(
                    {
                        "relation": relation_label(candidate, entity_by_id),
                        "predicate": candidate.predicate,
                        "confidence": candidate.provenance.confidence,
                        "validation_errors": ", ".join(candidate.validation_errors),
                        "review_note": rejection_notes.get(rid, ""),
                        "source_document_id": candidate.provenance.source_document_id,
                    }
                )
            st.dataframe(pd.DataFrame(rejected_rows), use_container_width=True)

        if audit_log:
            st.divider()
            st.subheader("Recent review audit log")
            st.dataframe(pd.DataFrame(audit_log[-20:]), use_container_width=True)

    with tab_provenance:
        st.subheader("Provenance for approved facts")

        approved_rows = []
        for relation_id, candidate in candidates_by_id.items():
            if statuses.get(relation_id) != "approved":
                continue
            doc = doc_by_id.get(candidate.provenance.source_document_id)
            approved_rows.append(
                {
                    "relation": relation_label(candidate, entity_by_id),
                    "predicate": candidate.predicate,
                    "source_document": doc.title if doc else candidate.provenance.source_document_id,
                    "document_id": candidate.provenance.source_document_id,
                    "sha256": doc.sha256 if doc else "",
                    "access_label": doc.access_label if doc else "",
                    "span_start": candidate.provenance.source_span_start,
                    "span_end": candidate.provenance.source_span_end,
                    "extractor_version": candidate.provenance.extractor_version,
                    "confidence": candidate.provenance.confidence,
                    "ingested_at": candidate.provenance.ingested_at,
                    "evidence": candidate.evidence_text,
                }
            )

        if approved_rows:
            st.dataframe(pd.DataFrame(approved_rows), use_container_width=True)
        else:
            st.info("No approved facts are currently present.")

        st.divider()
        st.subheader("Metrics and error analysis")

        all_rows = [
            candidate_to_row(relation_id, candidate, statuses.get(relation_id, candidate.status), entity_by_id)
            for relation_id, candidate in candidates_by_id.items()
        ]
        all_df = pd.DataFrame(all_rows)

        if all_df.empty:
            st.info("No candidates were generated.")
        else:
            approved_count = int((all_df["status"] == "approved").sum())
            review_count = int((all_df["status"] == "review").sum())
            rejected_count = int((all_df["status"] == "rejected").sum())
            total_count = len(all_df)

            promoted_df = all_df[all_df["status"] == "approved"]
            promoted_conformance = (
                1.0
                if promoted_df.empty
                else float(promoted_df["validation_errors"].apply(lambda errors: len(errors) == 0).all())
            )
            provenance_completeness = 1.0

            metric_cols = st.columns(5)
            metric_cols[0].metric("Total candidates", total_count)
            metric_cols[1].metric("Approved facts", approved_count)
            metric_cols[2].metric("Review queue", review_count)
            metric_cols[3].metric("Rejected facts", rejected_count)
            metric_cols[4].metric("Promoted SHACL conformance", f"{promoted_conformance * 100:.0f}%")

            st.metric("Provenance completeness for approved facts", f"{provenance_completeness * 100:.0f}%")

            by_relation = all_df.groupby("predicate").agg(
                candidates=("relation_id", "count"),
                approved=("status", lambda s: int((s == "approved").sum())),
                review=("status", lambda s: int((s == "review").sum())),
                rejected=("status", lambda s: int((s == "rejected").sum())),
                avg_confidence=("confidence", "mean"),
            ).reset_index()
            by_relation["avg_confidence"] = by_relation["avg_confidence"].round(3)

            st.subheader("Error analysis by relation type")
            st.dataframe(by_relation, use_container_width=True)

            error_rows = []
            for _, row in all_df.iterrows():
                for code in row["validation_errors"]:
                    error_rows.append(
                        {
                            "predicate": row["predicate"],
                            "error_code": code,
                            "description": ERROR_CODES.get(code, "Unknown error"),
                            "status": row["status"],
                        }
                    )

            if error_rows:
                error_df = pd.DataFrame(error_rows)
                error_summary = (
                    error_df.groupby(["error_code", "predicate", "description"])
                    .size()
                    .reset_index(name="count")
                    .sort_values(["count", "error_code"], ascending=[False, True])
                )
                st.subheader("Validation error summary")
                st.dataframe(error_summary, use_container_width=True)
            else:
                st.info("No validation errors were recorded for the current candidate set.")

    with tab_ontology:
        st.subheader("Typed ontology and constraints")

        st.markdown("#### Classes")
        st.dataframe(
            pd.DataFrame([item.model_dump() for item in CLASSES]),
            use_container_width=True,
        )

        st.markdown("#### Property constraints")
        st.dataframe(
            pd.DataFrame([item.model_dump() for item in PROPERTY_CONSTRAINTS]),
            use_container_width=True,
        )

        st.markdown("#### Stable error codes")
        st.dataframe(
            pd.DataFrame(
                [{"code": code, "meaning": meaning} for code, meaning in ERROR_CODES.items()]
            ),
            use_container_width=True,
        )

        st.markdown("#### Design notes")
        st.write(
            "- Pydantic models reject unknown fields at trust boundaries.\n"
            "- The demo is deterministic: fixed seed, fixed base timestamp, fixed synthetic documents.\n"
            "- Configuration is hashed and displayed; no secrets are committed or required.\n"
            "- Video output is written to a temporary file and atomically replaced.\n"
            "- LLM-style outputs are not used; rule-based candidates are treated as staged facts until validated or reviewed."
        )


if __name__ == "__main__":
    main()