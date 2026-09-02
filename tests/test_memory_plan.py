"""Planning a scan against the memory a card actually has."""

import pytest

from dcc_gliner_api.services.memory_plan import (
    ActivationCost,
    ScanPlan,
    derived_cost,
    plan_scan,
    profile_cost,
    split_labels,
)

GIB = 1024**3

#: What the model was weighed at: 360 bytes per pair of tokens, per chunk.
COST = ActivationCost(bytes_per_token_squared=360, source="test")


def test_a_small_schema_uses_the_batch_the_caller_asked_for():
    plan = plan_scan(budget_bytes=8 * GIB, sequence_length=1017, wanted_batch=8, label_count=2, cost=COST)
    assert plan == ScanPlan(batch_size=8, schema_groups=1)


def test_a_long_sequence_shrinks_the_batch_rather_than_the_schema():
    """The measured case: a 27-label schema on the card that ran out of memory."""
    plan = plan_scan(budget_bytes=8 * GIB, sequence_length=3142, wanted_batch=8, label_count=27, cost=COST)

    assert plan.schema_groups == 1
    assert plan.batch_size == 2
    assert COST.memory_for(plan.batch_size, 3142) <= 8 * GIB


def test_a_schema_that_does_not_fit_at_all_is_split():
    plan = plan_scan(budget_bytes=GIB // 2, sequence_length=6000, wanted_batch=8, label_count=40, cost=COST)

    assert plan.batch_size == 1
    assert plan.schema_groups > 1


def test_the_split_never_asks_for_more_groups_than_there_are_labels():
    plan = plan_scan(budget_bytes=1024, sequence_length=9000, wanted_batch=8, label_count=3, cost=COST)
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


def test_the_cost_is_the_growth_between_two_passes_not_their_size():
    """A fixed overhead in both passes must not be charged to the sequence."""
    overhead = 500 * 1024**2

    def probe(words: int) -> tuple[int, int]:
        length = 500 if words < 100 else 1500
        return overhead + 360 * length * length, length

    cost = profile_cost(probe, fallback=derived_cost(12, 4))

    assert cost.source == "profiled"
    assert cost.bytes_per_token_squared == 360


def test_a_replica_that_cannot_weigh_itself_falls_back_to_its_shape():
    def probe(words: int) -> tuple[int, int]:
        raise RuntimeError("no GPU to weigh")

    cost = profile_cost(probe, fallback=derived_cost(num_heads=12, bytes_per_element=4))

    assert cost.source == "derived"
    assert cost.bytes_per_token_squared == 8 * 12 * 4


def test_half_precision_halves_the_derived_cost():
    assert derived_cost(12, 2).bytes_per_token_squared == derived_cost(12, 4).bytes_per_token_squared // 2


def test_a_probe_that_measures_no_growth_is_not_believed():
    fallback = derived_cost(12, 4)
    assert profile_cost(lambda words: (1_000_000, 1000), fallback=fallback) == fallback
