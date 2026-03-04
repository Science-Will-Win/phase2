"""
Standalone preprocessing script for agent training data.

Converts raw agent results (train_biological_casuality_*_results_*.json)
into training-ready format with:
  - Tag conversion: <think> -> [THINK], <execute> -> [EXECUTE], etc.
  - Role mapping:   system -> system, human -> user, LLM -> assistant, Result -> tool
  - Error filtering: removes instances with status="error", empty final_answer,
                     or empty LLM content turns
  - Output:         JSON array with {"messages": [...]} per instance

Usage:
    python data_formatting/format_agent_data.py input1.json [input2.json ...] -o output.json
"""

import argparse
import json
import os
import re
import sys


TAG_MAP = {
    "<think>":          "[THINK]",
    "</think>":         "[/THINK]",
    "<execute>":        "[EXECUTE]",
    "</execute>":       "[/EXECUTE]",
    "<observation>":    "[OBSERVATION]",
    "</observation>":   "[/OBSERVATION]",
    "<solution>":       "[SOLUTION]",
    "</solution>":      "[/SOLUTION]",
}

TAG_PATTERN = re.compile(
    "|".join(re.escape(k) for k in TAG_MAP),
    flags=re.IGNORECASE,
)

ROLE_MAP = {
    "system":  "system",
    "human":   "user",
    "LLM":     "assistant",
    "Result":  "tool",
}


def convert_tags(text: str) -> str:
    """Replace agent-specific XML tags with Ministral bracket tokens."""
    return TAG_PATTERN.sub(lambda m: TAG_MAP[m.group(0).lower()], text)


def convert_instance(instance: dict) -> dict | None:
    """Convert a single raw instance to training format.

    Returns None if the instance should be filtered out.
    """
    if instance.get("status") == "error":
        return None

    final_answer = instance.get("final_answer", "")
    if not final_answer or not final_answer.strip():
        return None

    raw_messages = instance.get("messages", [])
    if not raw_messages:
        return None

    messages = []
    for msg in raw_messages:
        raw_type = msg.get("type", "")
        role = ROLE_MAP.get(raw_type)
        if role is None:
            continue

        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        if role == "assistant" and not content.strip():
            continue

        content = convert_tags(content)
        messages.append({"role": role, "content": content})

    if len(messages) < 2:
        return None

    result = {
        "instance_id": instance.get("instance_id"),
        "task_instance_id": instance.get("task_instance_id"),
        "messages": messages,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw agent results into training-ready format."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Input JSON file(s) (raw agent results)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    all_instances = []
    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"Warning: file not found, skipping: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        all_instances.extend(data)

    print(f"Loaded {len(all_instances)} raw instances from {len(args.inputs)} file(s)")

    converted = []
    skipped = 0
    for inst in all_instances:
        result = convert_instance(inst)
        if result is not None:
            converted.append(result)
        else:
            skipped += 1

    print(f"Converted: {len(converted)}, Filtered out: {skipped}")

    if not converted:
        print("No instances remaining after filtering. Exiting.")
        sys.exit(1)

    role_counts = {}
    tag_counts = {}
    for item in converted:
        for msg in item["messages"]:
            role_counts[msg["role"]] = role_counts.get(msg["role"], 0) + 1
            for tag in ["[THINK]", "[EXECUTE]", "[OBSERVATION]", "[SOLUTION]"]:
                if tag in msg["content"]:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print(f"\nRole distribution: {role_counts}")
    print(f"Tag occurrences:   {tag_counts}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {args.output} ({os.path.getsize(args.output):,} bytes)")


if __name__ == "__main__":
    main()
