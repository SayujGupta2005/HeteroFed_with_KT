import time
import json
from pathlib import Path
from typing import Dict, List

class TimeProfiler:
    def __init__(self):
        self.events: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}

    def start(self, event_name: str):
        self.start_times[event_name] = time.perf_counter()

    def stop(self, event_name: str):
        if event_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[event_name]
            if event_name not in self.events:
                self.events[event_name] = []
            self.events[event_name].append(elapsed)

    def save_summary(self, filepath: Path):
        summary = {}
        total_time = 0.0
        for event, times in self.events.items():
            tot = sum(times)
            total_time += tot
            summary[event] = {
                "total_time_sec": round(tot, 2),
                "avg_time_sec": round(tot / len(times), 2),
                "calls": len(times)
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
        
        txt_path = filepath.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("       EXECUTION TIME SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            for event, stats in summary.items():
                f.write(f"[{event}]\n")
                f.write(f"  Total Time : {stats['total_time_sec']:.2f} s\n")
                f.write(f"  Avg Time   : {stats['avg_time_sec']:.2f} s\n")
                f.write(f"  Calls      : {stats['calls']}\n\n")
                
global_profiler = TimeProfiler()

