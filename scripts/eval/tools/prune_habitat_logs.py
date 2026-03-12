import argparse
import os
import shutil
from pathlib import Path


DEFAULT_PATTERNS = [
    "replay_subset",
    "replay_subset_v2",
    "vis_0",
    "vis_debug",
    "check_sim_0",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prune bulky Habitat log artifacts while keeping summary files.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root log directory to scan, e.g. /root/backup/InternNav/logs/habitat/closed_loop_num_history_sweeps",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=DEFAULT_PATTERNS,
        help="Directory names to prune.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matched directories. Default is dry-run.",
    )
    return parser.parse_args()


def sizeof_path(path):
    total = 0
    if path.is_file():
        return path.stat().st_size
    for root, _, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            if os.path.exists(full):
                total += os.path.getsize(full)
    return total


def format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Log root does not exist: {root}")

    matches = []
    for current_root, dirnames, _ in os.walk(root):
        for dirname in dirnames:
            if dirname in args.patterns:
                path = Path(current_root) / dirname
                matches.append(path)

    matches = sorted(set(matches))
    total_bytes = 0
    for path in matches:
        size = sizeof_path(path)
        total_bytes += size
        action = "DELETE" if args.apply else "WOULD_DELETE"
        print(f"{action}\t{format_bytes(size)}\t{path}")

    print(f"TOTAL\t{format_bytes(total_bytes)}")

    if args.apply:
        for path in matches:
            shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
