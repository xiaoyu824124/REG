"""Group the existing HDO IR/VI pairs without modifying the flat originals.

The output is E:/HDO/{aligned,RAW}/hdoN/{ir,vi}. Files are NTFS hard links
to the originals, so this does not duplicate the image bytes. Aligned letters
map a->1, ..., z->26; RAW uses the numeric prefix in the filename.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


PATTERNS = {
    "aligned": re.compile(r"^([a-z])(\d+)\.jpg$", re.IGNORECASE),
    "RAW": re.compile(r"^(\d+) \((\d+)\)\.jpg$", re.IGNORECASE),
}


def parse_group(kind: str, name: str) -> tuple[int, int] | None:
    match = PATTERNS[kind].fullmatch(name)
    if not match:
        return None
    group = ord(match.group(1).lower()) - ord("a") + 1 if kind == "aligned" else int(match.group(1))
    return group, int(match.group(2))


def inventory(root: Path, kind: str) -> tuple[dict[int, list[str]], dict[str, list[str]]]:
    source = root / kind
    if not source.is_dir():
        raise FileNotFoundError(source)
    names: dict[str, dict[str, Path]] = {}
    invalid: dict[str, list[str]] = {}
    for modality in ("ir", "vi"):
        folder = source / modality
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files = [entry for entry in folder.iterdir() if entry.is_file()]
        names[modality] = {entry.name: entry for entry in files if parse_group(kind, entry.name)}
        invalid[modality] = sorted(entry.name for entry in files if not parse_group(kind, entry.name))

    paired = names["ir"].keys() & names["vi"].keys()
    invalid["ir"].extend(sorted(names["ir"].keys() - paired))
    invalid["vi"].extend(sorted(names["vi"].keys() - paired))
    grouped: dict[int, list[str]] = defaultdict(list)
    for name in paired:
        group, _ = parse_group(kind, name)  # type: ignore[misc]
        grouped[group].append(name)
    for group, items in grouped.items():
        items.sort(key=lambda name: parse_group(kind, name)[1])  # type: ignore[index]
    return dict(grouped), invalid


def link_checked(source: Path, destination: Path) -> str:
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise FileExistsError(f"Existing file is not a link to its source: {destination}")
        return "existing"
    os.link(source, destination)
    if not os.path.samefile(source, destination):
        raise RuntimeError(f"Hard-link verification failed: {destination}")
    return "created"


def organize(root: Path) -> dict:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    inventories = {kind: inventory(root, kind) for kind in PATTERNS}
    manifest: dict = {
        "root": str(root),
        "layout": "{aligned,RAW}/hdoN/{ir,vi}/original_filename.jpg",
        "method": "NTFS hard links; flat originals preserved",
        "datasets": {},
    }
    for kind in ("aligned", "RAW"):
        grouped, excluded = inventories[kind]
        created = 0
        reused = 0
        groups = {}
        for group in sorted(grouped):
            names = grouped[group]
            base = root / kind / f"hdo{group}"
            for modality in ("ir", "vi"):
                (base / modality).mkdir(parents=True, exist_ok=True)
            for name in names:
                for modality in ("ir", "vi"):
                    status = link_checked(root / kind / modality / name, base / modality / name)
                    created += status == "created"
                    reused += status == "existing"
            groups[f"hdo{group}"] = {
                "pairs": len(names),
                "first": names[0],
                "last": names[-1],
            }
            print(f"{kind}/hdo{group}: {len(names)} IR/VI pairs", flush=True)
        manifest["datasets"][kind] = {
            "groups": groups,
            "total_pairs": sum(item["pairs"] for item in groups.values()),
            "linked_files_created": created,
            "linked_files_reused": reused,
            "excluded_unpaired_or_noncanonical": excluded,
        }

    # A completed manifest is written only after every pair has been linked.
    path = root / "hdo_grouping_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"E:\HDO"))
    args = parser.parse_args()
    manifest = organize(args.root)
    for kind, summary in manifest["datasets"].items():
        print(
            f"DONE {kind}: {len(summary['groups'])} groups, "
            f"{summary['total_pairs']} pairs, "
            f"{len(summary['excluded_unpaired_or_noncanonical']['ir'])} extra IR, "
            f"{len(summary['excluded_unpaired_or_noncanonical']['vi'])} extra VI",
            flush=True,
        )


if __name__ == "__main__":
    main()
