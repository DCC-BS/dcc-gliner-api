"""Planning a scan against the memory a card actually has."""

import pytest

from dcc_gliner_api.services.memory_plan import (
    ScanPlan,
    memory_for,
    plan_scan,
    split_labels,
)

GIB = 1024**3


def test_a_small_schema_uses_the_batch_the_caller_asked_for():
    plan = plan_scan(budget_bytes=8 * GIB, sequence_length=1017, wanted_batch=8, label_count=2)
    assert plan == ScanPlan(batch_size=8, schema_groups=1)


def test_a_long_sequence_shrinks_the_batch_rather_than_the_schema():
    """The measured case: a 27-label schema on the card that ran out of memory."""
    plan = plan_scan(budget_bytes=8 * GIB, sequence_length=3142, wanted_batch=8, label_count=27)

    assert plan.schema_groups == 1
    assert plan.batch_size == 2
    assert memory_for(plan.batch_size, 3142) <= 8 * GIB


def test_a_schema_that_does_not_fit_at_all_is_split():
    plan = plan_scan(budget_bytes=GIB // 2, sequence_length=6000, wanted_batch=8, label_count=40)

    assert plan.batch_size == 1
    assert plan.schema_groups > 1


def test_the_split_never_asks_for_more_groups_than_there_are_labels():
    plan = plan_scan(budget_bytes=1024, sequence_length=9000, wanted_batch=8, label_count=3)
    assert plan.schema_groups == 3


@pytest.mark.parametrize("groups", [2, 3, 4])
def test_splitting_keeps_every_label_exactly_once(groups):
    schema = {f"label_{index}": f"description {index}" for index in range(9)}
    pieces = split_labels(schema, groups)

    assert len(pieces) == groups
    seen = [label for piece in pieces for label in piece]
    assert sorted(seen) == sorted(schema)
    assert all(piece[label] == schema[label] for piece in pieces for label in piece)


def test_a_schema_given_as_a_list_splits_as_a_list():
    pieces = split_labels(["a", "b", "c", "d"], 2)
    assert pieces == [["a", "b"], ["c", "d"]]


def test_one_group_leaves_the_schema_alone():
    schema = {"person": "a person"}
    assert split_labels(schema, 1) == [schema]
