"""Deterministic train/validation/test splitting for adjacent flight frames."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SequenceGroup:
    stems: tuple[str, ...]
    new_count: int

    @property
    def size(self) -> int:
        return len(self.stems)


def _frame_key(stem: str) -> tuple[str, int] | None:
    match = re.match(r"^(.*?)(\d+)$", stem)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _ratio_targets(total: int, ratios: Sequence[float]) -> list[int]:
    raw = [total * ratio / sum(ratios) for ratio in ratios]
    targets = [int(value) for value in raw]
    for index in sorted(range(len(raw)), key=lambda item: raw[item] - targets[item], reverse=True)[: total - sum(targets)]:
        targets[index] += 1
    return targets


def build_sequence_groups(
    stems: Sequence[str],
    is_new: Mapping[str, bool],
    max_gap: int = 10,
) -> list[SequenceGroup]:
    numbered: list[tuple[str, int, str]] = []
    unnumbered: list[str] = []
    for stem in stems:
        key = _frame_key(stem)
        if key is None:
            unnumbered.append(stem)
        else:
            numbered.append((key[0], key[1], stem))
    numbered.sort()

    groups: list[list[str]] = []
    current: list[str] = []
    previous_prefix: str | None = None
    previous_number: int | None = None
    for prefix, number, stem in numbered:
        if current and (prefix != previous_prefix or number - int(previous_number) > max_gap):
            groups.append(current)
            current = []
        current.append(stem)
        previous_prefix, previous_number = prefix, number
    if current:
        groups.append(current)
    groups.extend([[stem] for stem in sorted(unnumbered)])
    return [SequenceGroup(tuple(group), sum(bool(is_new.get(stem)) for stem in group)) for group in groups]


def _choose_subset(
    groups: Sequence[SequenceGroup],
    target_size: int,
    target_new: int,
    rng: random.Random,
) -> set[int] | None:
    order = list(range(len(groups)))
    rng.shuffle(order)
    states: dict[tuple[int, int], int] = {(0, 0): 0}
    for index in order:
        group = groups[index]
        additions: dict[tuple[int, int], int] = {}
        for (size, new_count), mask in states.items():
            next_size = size + group.size
            if next_size > target_size:
                continue
            key = (next_size, new_count + group.new_count)
            if key not in states and key not in additions:
                additions[key] = mask | (1 << index)
        states.update(additions)

    candidates = [(key, mask) for key, mask in states.items() if key[0] == target_size]
    if not candidates:
        return None
    best_difference = min(abs(key[1] - target_new) for key, _ in candidates)
    best = [(key, mask) for key, mask in candidates if abs(key[1] - target_new) == best_difference]
    _, mask = rng.choice(best)
    return {index for index in range(len(groups)) if mask & (1 << index)}


def grouped_split(
    stems: Sequence[str],
    is_new: Mapping[str, bool],
    ratios: Sequence[float] = (0.7, 0.2, 0.1),
    max_gap: int = 10,
    seed: int = 42,
    attempts: int = 5000,
) -> dict[str, list[str]]:
    """Split stems while keeping adjacent frames together and balancing sources."""
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must contain three positive values")
    unique_stems = sorted(set(stems))
    if len(unique_stems) != len(stems):
        raise ValueError("stems must be unique")

    names = ("train", "valid", "test")
    size_targets = dict(zip(names, _ratio_targets(len(unique_stems), ratios)))
    total_new = sum(bool(is_new.get(stem)) for stem in unique_stems)
    new_targets = dict(zip(names, _ratio_targets(total_new, ratios)))
    groups = build_sequence_groups(unique_stems, is_new, max_gap=max_gap)

    rng = random.Random(seed)
    best: tuple[int, dict[str, list[str]]] | None = None
    for attempt in range(attempts):
        first, second = (("test", "valid") if attempt % 2 == 0 else ("valid", "test"))
        first_indices = _choose_subset(groups, size_targets[first], new_targets[first], rng)
        if first_indices is None:
            continue
        remaining_indices = [index for index in range(len(groups)) if index not in first_indices]
        remaining_groups = [groups[index] for index in remaining_indices]
        second_local = _choose_subset(remaining_groups, size_targets[second], new_targets[second], rng)
        if second_local is None:
            continue
        second_indices = {remaining_indices[index] for index in second_local}

        assigned = {first: first_indices, second: second_indices}
        assigned["train"] = set(range(len(groups))) - first_indices - second_indices
        result = {
            name: sorted(stem for index in assigned[name] for stem in groups[index].stems)
            for name in names
        }
        score = sum(
            abs(sum(bool(is_new.get(stem)) for stem in result[name]) - new_targets[name])
            for name in names
        )
        if best is None or score < best[0]:
            best = score, result
            if score == 0:
                break

    if best is None:
        sizes = sorted((group.size for group in groups), reverse=True)
        raise ValueError(f"Cannot satisfy split sizes {size_targets} with sequence-group sizes {sizes}")
    return best[1]


def adjacent_cross_split_pairs(splits: Mapping[str, Sequence[str]], max_gap: int = 10) -> list[tuple[str, str]]:
    locations = {stem: split for split, stems in splits.items() for stem in stems}
    numbered = [(key[0], key[1], stem) for stem in locations if (key := _frame_key(stem)) is not None]
    numbered.sort()
    crossed: list[tuple[str, str]] = []
    for previous, current in zip(numbered, numbered[1:]):
        if previous[0] == current[0] and current[1] - previous[1] <= max_gap:
            if locations[previous[2]] != locations[current[2]]:
                crossed.append((previous[2], current[2]))
    return crossed
