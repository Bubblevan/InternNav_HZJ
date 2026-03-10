import argparse
import gzip
import json
import os
import random
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description="Build a tiny R2R json.gz subset for Habitat eval.")
    parser.add_argument("--input", required=True, help="Path to source json/json.gz")
    parser.add_argument("--output", required=True, help="Path to output json.gz")
    parser.add_argument("--max-episodes", type=int, default=8, help="Maximum number of episodes to keep")
    parser.add_argument("--scene-id", type=str, default=None, help="Optional scene id to keep")
    parser.add_argument("--episode-ids", nargs="*", type=int, default=None, help="Optional explicit episode ids")
    parser.add_argument("--seed", type=int, default=0, help="Random seed when sampling")
    parser.add_argument("--sample-random", action="store_true", help="Randomly sample instead of taking the first N")
    return parser.parse_args()


def load_payload(path: str) -> dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_payload(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def main():
    args = parse_args()
    payload = load_payload(args.input)
    episodes = payload["episodes"]

    if args.scene_id:
        episodes = [ep for ep in episodes if ep["scene_id"].split("/")[-2] == args.scene_id]

    if args.episode_ids:
        keep_ids = {int(ep_id) for ep_id in args.episode_ids}
        episodes = [ep for ep in episodes if int(ep["episode_id"]) in keep_ids]

    if args.sample_random:
        rng = random.Random(args.seed)
        rng.shuffle(episodes)

    episodes = episodes[: args.max_episodes]
    payload["episodes"] = episodes
    dump_payload(args.output, payload)

    scene_ids = sorted({ep["scene_id"].split("/")[-2] for ep in episodes})
    print(f"saved {len(episodes)} episodes to {args.output}")
    print(f"scene_ids={scene_ids}")
    print(f"episode_ids={[int(ep['episode_id']) for ep in episodes]}")


if __name__ == "__main__":
    main()
