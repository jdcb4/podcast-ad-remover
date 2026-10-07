"""Conservative grouping of overlapping transcript segments without word alignment."""
from collections.abc import Iterator


def overlap_groups(segments: list[dict]) -> Iterator[list[dict]]:
    """Yield connected overlap groups from segments in chronological order."""
    group, end = [], 0.0
    for segment in segments:
        if group and segment['start'] >= end - 1e-7:
            yield group
            group = []
            end = 0.0
        group.append(segment)
        end = max(end, segment['end'])
    if group:
        yield group


def combine_segments(segments: list[dict]) -> dict:
    """Keep the full interval and every text alternative as one indivisible item."""
    return {**segments[0], 'start': min(s['start'] for s in segments),
            'end': max(s['end'] for s in segments),
            'text': '\n'.join(s['text'].strip() for s in segments if s['text'].strip())}


def merge_chunk_segments(segments: list[dict]) -> list[dict]:
    """Coalesce cross-chunk overlaps; keep same-chunk segment boundaries intact."""
    merged = []
    for group in overlap_groups(sorted(segments, key=lambda s: s['start'])):
        if len({s['chunk_index'] for s in group}) > 1:
            combined = combine_segments(group)
            # Retain the actual timestamps/text for diagnostics. Do not invent
            # word-level boundaries or discard a shorter alternative as a duplicate.
            combined['source_segments'] = [dict(s) for s in group]
            merged.append(combined)
        else:
            merged.extend(dict(s) for s in group)
    for i, segment in enumerate(merged):
        segment['id'] = i
    return merged
