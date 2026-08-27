"""Run-summary reporting for the ingestion pipeline.

`main.py` prints a human-readable breakdown of what the cleaning pipeline did:
how many rows entered each stage, how many were removed, and why (platform /
media-type filters, fully-blank rows, duplicate links, minimum-engagement cut).
This answers "how many were removed by de-duplication" (and every other stage)
at a glance.

The summary is a plain dict so it can also be returned programmatically or
serialised to JSON without pulling in heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STAGE_ORDER = (
    "input",
    "after_platform_filter",
    "after_media_type_filter",
    "after_type_clean",
    "after_drop_blank",
    "after_dedupe",
    "after_min_rows",
    "output",
)


@dataclass
class CleanSummary:
    """Row counts captured at each cleaning stage.

    ``counts`` is keyed by the stage names in ``STAGE_ORDER``. ``dropped`` is
    derived: for each stage it holds how many rows were removed between the
    previous and current stage.
    """

    counts: dict[str, int] = field(default_factory=dict)
    platforms: list[str] = field(default_factory=list)
    media_types: list[str] = field(default_factory=list)
    dedupe_enabled: bool = True
    min_full_rows: int = 0

    @property
    def dropped(self) -> dict[str, int]:
        """Rows removed between consecutive stages (keyed by destination stage)."""
        result: dict[str, int] = {}
        stages = [s for s in STAGE_ORDER if s in self.counts]
        prev = None
        for s in stages:
            if prev is not None:
                result[s] = self.counts[prev] - self.counts[s]
            else:
                result[s] = 0
            prev = s
        return result

    def format(self) -> str:
        """Render the summary as a readable text block."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("INGESTION RUN SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Platforms kept  : {', '.join(self.platforms)}")
        lines.append(f"Media types kept: {', '.join(self.media_types)}")
        lines.append(f"Drop duplicates : {'yes' if self.dedupe_enabled else 'no'}")
        if self.min_full_rows:
            lines.append(f"Min engagement cols per row: {self.min_full_rows}")
        lines.append("-" * 60)
        lines.append(f"{'Input rows':<38}{self.counts.get('input', 0):>12,}")
        lines.append(f"{'  after platform filter':<38}{self.counts.get('after_platform_filter', 0):>12,}")
        lines.append(f"{'  after media-type filter':<38}{self.counts.get('after_media_type_filter', 0):>12,}")
        lines.append(f"{'  after type cleaning':<38}{self.counts.get('after_type_clean', 0):>12,}")
        lines.append(f"{'  after dropping blank URL+content rows':<38}{self.counts.get('after_drop_blank', 0):>12,}")
        lines.append(f"{'  after removing duplicate links':<38}{self.counts.get('after_dedupe', 0):>12,}")
        if self.min_full_rows:
            lines.append(f"{'  after minimum-engagement cut':<38}{self.counts.get('after_min_rows', 0):>12,}")
        lines.append(f"{'Output rows':<38}{self.counts.get('output', 0):>12,}")
        lines.append("-" * 60)
        lines.append("Rows removed by stage:")
        for stage in STAGE_ORDER:
            if stage == "input":
                continue
            if stage in self.dropped and self.dropped[stage] > 0:
                lines.append(f"  - {stage:<32}{self.dropped[stage]:>10,} removed")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "counts": self.counts,
            "dropped": self.dropped,
            "platforms": self.platforms,
            "media_types": self.media_types,
            "dedupe_enabled": self.dedupe_enabled,
            "min_full_rows": self.min_full_rows,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CleanSummary":
        """Rebuild a summary from a ``to_json()`` dict (e.g. recovered from df.attrs)."""
        return cls(
            counts=dict(data.get("counts", {})),
            platforms=list(data.get("platforms", [])),
            media_types=list(data.get("media_types", [])),
            dedupe_enabled=bool(data.get("dedupe_enabled", True)),
            min_full_rows=int(data.get("min_full_rows", 0)),
        )
