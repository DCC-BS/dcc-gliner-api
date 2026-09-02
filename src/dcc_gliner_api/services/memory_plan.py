"""How much of a scan fits on the GPU at once.

Attention over a sequence costs memory proportional to its square, and the
sequence a chunk is scanned in is the schema plus the chunk: every label and
its description is read alongside every chunk. A schema of two dozen described
labels is therefore not a little more expensive than two labels — it is orders
of magnitude more, and a batch of chunks multiplies it.

So nothing here is fixed in advance. The budget comes from the card the model
actually landed on, the sequence is measured for the schema in hand, and the
two decide how many chunks may go into one model call — and, when even a
single chunk will not fit, into how many groups the schema has to be split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Activation bytes per chunk per token squared.
#:
#: Measured on an RTX 5090 with mdeberta-v3-base: peak allocation minus the
#: 1.14 GiB of weights, over batch x sequence^2, came to 267-360 bytes across
#: sequences of 1017 to 3142 tokens and batches of 1 to 8. The largest is taken,
#: since underestimating costs an out-of-memory error and overestimating only
#: costs a smaller batch.
BYTES_PER_CHUNK_TOKEN_SQUARED = 360

#: Share of the free memory a scan may claim, leaving room for fragmentation
#: and for whatever else shares the card.
DEFAULT_SAFETY_MARGIN = 0.8


@dataclass(frozen=True)
class ScanPlan:
    """How to run one scan within the budget."""

    #: Chunks per model call.
    batch_size: int
    #: How many pieces the schema must be split into; 1 leaves it whole.
    schema_groups: int


def memory_for(batch_size: int, sequence_length: int) -> int:
    """Activation bytes one model call is expected to need."""
    return BYTES_PER_CHUNK_TOKEN_SQUARED * batch_size * sequence_length * sequence_length


def plan_scan(budget_bytes: int, sequence_length: int, wanted_batch: int, label_count: int) -> ScanPlan:
    """Work out the largest batch that fits, splitting the schema if none does.

    Args:
        budget_bytes: Activation memory this scan may use.
        sequence_length: Tokens in one chunk's input, schema included.
        wanted_batch: Chunks the caller would like to send at once.
        label_count: Labels in the schema, an upper bound on the split.

    Returns:
        The batch size to use, and the number of groups to split the schema
        into. A plan is always returned: a single label at a single chunk is
        the smallest possible scan, and the caller runs it whichever way the
        arithmetic came out.
    """
    fits = budget_bytes // memory_for(1, sequence_length)

    if fits >= 1:
        return ScanPlan(batch_size=min(wanted_batch, int(fits)), schema_groups=1)

    # One chunk does not fit, so the schema itself is the problem. Its share of
    # the sequence shrinks with each group; the chunk's own tokens do not, which
    # is why this solves for the ratio rather than dividing the length.
    needed = memory_for(1, sequence_length) / max(budget_bytes, 1)
    groups = min(label_count, max(2, math.ceil(math.sqrt(needed))))

    return ScanPlan(batch_size=1, schema_groups=groups)


def split_labels(entity_types: dict[str, str] | list[str], groups: int) -> list[dict[str, str] | list[str]]:
    """Divide a schema into ``groups`` roughly equal pieces, in order.

    Order is kept rather than shuffled so that the same document scanned twice
    is scanned the same way, and a result can be traced back to the group that
    produced it.
    """
    if groups <= 1:
        return [entity_types]

    labels = list(entity_types)
    groups = min(groups, len(labels))

    # Exactly as many pieces as asked for, sizes differing by at most one: a
    # piece larger than planned is a piece that does not fit.
    size, remainder = divmod(len(labels), groups)
    pieces: list[list[str]] = []
    start = 0
    for index in range(groups):
        end = start + size + (1 if index < remainder else 0)
        pieces.append(labels[start:end])
        start = end

    if isinstance(entity_types, dict):
        return [{label: entity_types[label] for label in piece} for piece in pieces]

    return list(pieces)
