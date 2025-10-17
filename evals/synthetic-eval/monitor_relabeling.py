"""
Quick script to monitor relabeling progress.
"""
import json
import time
from pathlib import Path

output_dir = Path(__file__).parent / "output"
stats_file = output_dir / "generation_stats.json"

print("Monitoring relabeling progress...")
print("Press Ctrl+C to stop\n")

last_stats = None

try:
    while True:
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)

            if 'silver_labeling' in stats:
                sl = stats['silver_labeling']

                if last_stats != sl:
                    print(f"\n=== Silver Labeling Stats ===")
                    print(f"Total labels: {sl.get('total_labels', 0):,}")
                    print(f"Total queries: {sl.get('total_queries', 0)}")

                    dist = sl.get('label_distribution', {})
                    print(f"\nLabel distribution:")
                    for label, count in sorted(dist.items()):
                        pct = sl.get('label_percentages', {}).get(label, 0)
                        print(f"  Rel {label}: {count:,} ({pct:.2f}%)")

                    print(f"\nAvg relevant per query: {sl.get('avg_relevant_per_query', 0):.2f}")
                    print(f"Queries with no relevant: {sl.get('queries_with_no_relevant', 0)}")

                    last_stats = sl.copy()

        time.sleep(5)

except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")
