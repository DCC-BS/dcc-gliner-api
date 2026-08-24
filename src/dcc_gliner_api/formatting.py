"""Shape raw GLiNER2 extraction results into stable API responses.

All model calls run with ``format_results=False`` because the upstream
formatter dedupes values by lowercased text and silently drops duplicate
mentions of the same text at different positions. These helpers reproduce
the upstream response shapes without that dedup.
"""

from __future__ import annotations

from typing import Any

LabelConf = tuple[str, float]
RawResult = dict[str, Any]


def relation_names_of(relation_types: list[str] | dict[str, Any] | str) -> list[str]:
    if isinstance(relation_types, str):
        return [relation_types]
    return list(relation_types)


def classification_tasks_of(spec: dict[str, Any]) -> list[str]:
    return [c["task"] for c in spec.get("classifications", [])]


def shape_entity_result(raw: RawResult) -> dict[str, Any]:
    """{"entities": [{"person": [...]}]} -> {"person": [...]} (empty -> {})."""
    entities = raw.get("entities") or []
    if not entities:
        return {}
    return {name: (value if isinstance(value, (list, dict)) or value else None) for name, value in entities[0].items()}


def _shape_classification(value: LabelConf | list[LabelConf], include_confidence: bool) -> Any:
    if isinstance(value, list):
        return [{"label": label, "confidence": conf} if include_confidence else label for label, conf in value]
    label, conf = value
    return {"label": label, "confidence": conf} if include_confidence else label


def shape_classification_result(raw: RawResult, task_names: list[str], include_confidence: bool) -> dict[str, Any]:
    return {task: _shape_classification(raw[task], include_confidence) for task in task_names if task in raw}


def _shape_relation_instances(instances: list[Any]) -> list[Any]:
    return [list(inst) if isinstance(inst, tuple) else inst for inst in instances]


def shape_relation_result(raw: RawResult, relation_names: list[str]) -> dict[str, Any]:
    extraction = {name: _shape_relation_instances(raw.get(name, [])) for name in relation_names}
    return {"relation_extraction": extraction} if extraction else {}


def shape_extraction_result(
    raw: RawResult,
    include_confidence: bool,
    classification_tasks: list[str] | None = None,
    relation_names: list[str] | None = None,
) -> dict[str, Any]:
    """Shape a mixed multi-task result (entities, classifications, relations, structures)."""
    classification_tasks = classification_tasks or []
    relation_names = relation_names or []

    formatted: dict[str, Any] = {}
    relations: dict[str, Any] = {}

    for key, value in raw.items():
        if key in classification_tasks:
            formatted[key] = _shape_classification(value, include_confidence)
        elif key in relation_names:
            relations[key] = _shape_relation_instances(value)
        elif key == "entities":
            formatted[key] = shape_entity_result(raw)
        else:
            formatted[key] = value

    for name in relation_names:
        relations.setdefault(name, [])

    if relations:
        formatted["relation_extraction"] = relations

    return formatted
