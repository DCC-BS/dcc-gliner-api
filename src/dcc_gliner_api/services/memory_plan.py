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

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("ray.serve")

#: Score tensors the attention keeps alive at once.
#:
#: DeBERTa's disentangled attention holds several [batch*heads, seq, seq]
#: tensors together — the content scores, the two relative-position terms with
#: their gathered copies, the accumulator and the softmax output. Counting them
#: from the forward pass gives the cost per head; it is only used when the
#: replica cannot weigh itself, and it is deliberately generous.
LIVE_SCORE_TENSORS = 8

#: Words of text the two profiling passes run, short and long. Far enough
#: apart that the growth between them is the square term rather than noise.
PROBE_WORDS_SHORT = 64
PROBE_WORDS_LONG = 320

#: Share of the free memory a scan may claim, leaving room for fragmentation
#: and for whatever else shares the card.
DEFAULT_SAFETY_MARGIN = 0.8


@dataclass(frozen=True)
class ActivationCost:
    """What one chunk costs, per pair of tokens in its sequence.

    Attention compares every token with every other, so the cost of a chunk
    grows with the square of its sequence, and a batch multiplies it:

        bytes = cost x batch x sequence^2

    The number is weighed on the card at startup where that is possible, and
    derived from the model's own shape where it is not. ``source`` says which,
    so a log line can be believed or doubted accordingly.
    """

    bytes_per_token_squared: int
    source: str

    def memory_for(self, batch_size: int, sequence_length: int) -> int:
        """Activation bytes one model call is expected to need."""
        return self.bytes_per_token_squared * batch_size * sequence_length * sequence_length


def derived_cost(num_heads: int, bytes_per_element: int) -> ActivationCost:
    """The cost implied by the model's shape, for when it cannot be weighed."""
    return ActivationCost(
        bytes_per_token_squared=LIVE_SCORE_TENSORS * num_heads * bytes_per_element,
        source="derived",
    )


def profile_cost(probe: Callable[[int], tuple[int, int]], fallback: ActivationCost) -> ActivationCost:
    """Weigh two forward passes and read the cost off the difference.

    A single pass would fold in what the model spends regardless of length —
    embeddings, the counting head, the allocator's own blocks — and charge it
    to the sequence, which overstates the cost of every later scan. Two passes
    of different lengths separate the part that grows with the square from the
    part that does not:

        cost = (bytes_long - bytes_short) / (long^2 - short^2)

    Args:
        probe: Runs one chunk of about the requested word count and returns the
            activation bytes it peaked at, with the sequence length it ran.
        fallback: Used when the probes cannot run — on a CPU replica, or when a
            pass fails.

    Returns:
        The measured cost, or ``fallback`` when it could not be measured or
        came out implausible.
    """
    try:
        short_bytes, short_length = probe(PROBE_WORDS_SHORT)
        long_bytes, long_length = probe(PROBE_WORDS_LONG)
    except Exception:
        logger.warning("Could not profile activation memory; using the derived cost", exc_info=True)
        return fallback

    span = long_length * long_length - short_length * short_length
    growth = long_bytes - short_bytes
    if span <= 0 or growth <= 0:
        logger.warning("Activation profile came out flat; using the derived cost")
        return fallback

    return ActivationCost(bytes_per_token_squared=math.ceil(growth / span), source="profiled")


@dataclass(frozen=True)
class ScanPlan:
    """How to run one scan within the budget."""

    #: Chunks per model call.
    batch_size: int
    #: How many pieces the schema must be split into; 1 leaves it whole.
    schema_groups: int


def plan_scan(
    budget_bytes: int,
    sequence_length: int,
    wanted_batch: int,
    label_count: int,
    cost: ActivationCost,
) -> ScanPlan:
    """Work out the largest batch that fits, splitting the schema if none does.

    Args:
        budget_bytes: Activation memory this scan may use.
        sequence_length: Tokens in one chunk's input, schema included.
        wanted_batch: Chunks the caller would like to send at once.
        label_count: Labels in the schema, an upper bound on the split.
        cost: What a chunk costs per pair of tokens.

    Returns:
        The batch size to use, and the number of groups to split the schema
        into. A plan is always returned: a single label at a single chunk is
        the smallest possible scan, and the caller runs it whichever way the
        arithmetic came out.
    """
    fits = budget_bytes // cost.memory_for(1, sequence_length)

    if fits >= 1:
        return ScanPlan(batch_size=min(wanted_batch, int(fits)), schema_groups=1)

    # One chunk does not fit, so the schema itself is the problem. Its share of
    # the sequence shrinks with each group; the chunk's own tokens do not, which
    # is why this solves for the ratio rather than dividing the length.
    needed = cost.memory_for(1, sequence_length) / max(budget_bytes, 1)
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
