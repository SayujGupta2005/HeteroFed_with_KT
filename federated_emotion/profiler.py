"""Execution time profiling utility for tracking stages of federated emotion pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List


class TimeProfiler:
    """Records timing events across client training and server aggregation stages."""

    def __init__(self) -> None:
        self.events: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}

    def start(self, event_name: str) -> None:
        """Start recording time for an event."""
        self.start_times[event_name] = time.perf_counter()

    def stop(self, event_name: str) -> None:
        """Stop recording time for an event and append elapsed seconds."""
        if event_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[event_name]
            if event_name not in self.events:
                self.events[event_name] = []
            self.events[event_name].append(elapsed)

    def save_summary(self, filepath: Path) -> None:
        """Export timing summary as both JSON and readable TXT."""
        summary: Dict[str, Dict[str, float]] = {}
        total_time = 0.0
        for event, times in self.events.items():
            tot = sum(times)
            total_time += tot
            summary[event] = {
                "total_time_sec": round(tot, 2),
                "avg_time_sec": round(tot / len(times), 2),
                "calls": len(times),
            }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        txt_path = filepath.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("       EXECUTION TIME SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Cumulative Tracked Time: {total_time:.2f} s\n\n")
            for event, stats in summary.items():
                f.write(f"[{event}]\n")
                f.write(f"  Total Time : {stats['total_time_sec']:.2f} s\n")
                f.write(f"  Avg Time   : {stats['avg_time_sec']:.2f} s\n")
                f.write(f"  Calls      : {int(stats['calls'])}\n\n")


global_profiler = TimeProfiler()
