from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd


PATH_COLUMNS = ["image_path", "mask_path", "topology_cache_path"]


def _rebase_value(value: object, *, old_root: Path, new_root: Path) -> object:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return value
    path_text = str(value)
    try:
        path = Path(path_text)
    except TypeError:
        return value
    if not path.is_absolute():
        return value
    try:
        relative = path.relative_to(old_root)
    except ValueError:
        return value
    return str((new_root / relative).resolve())


def _iter_manifest_paths(inputs: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    for path in inputs:
        if path.is_dir():
            manifests.extend(sorted(path.rglob("*.csv")))
        elif path.suffix.lower() == ".csv":
            manifests.append(path)
        else:
            raise ValueError(f"Expected CSV file or directory, got: {path}")
    return sorted(set(manifests))


def _iter_json_paths(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.json"))
    if path.suffix.lower() == ".json":
        return [path]
    raise ValueError(f"Expected JSON file or directory, got: {path}")


def rebase_manifest(path: Path, *, old_root: Path, new_root: Path, dry_run: bool) -> bool:
    frame = pd.read_csv(path)
    changed = False
    for column in PATH_COLUMNS:
        if column not in frame.columns:
            continue
        updated = frame[column].map(
            lambda value: _rebase_value(value, old_root=old_root, new_root=new_root)
        )
        if not updated.equals(frame[column]):
            frame[column] = updated
            changed = True
    if changed and not dry_run:
        frame.to_csv(path, index=False)
    return changed


def rebase_topology_cache_json(
    path: Path,
    *,
    old_root: Path,
    new_root: Path,
    dry_run: bool,
) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "mask_path" not in data:
        return False
    updated = _rebase_value(data["mask_path"], old_root=old_root, new_root=new_root)
    if updated == data["mask_path"]:
        return False
    data["mask_path"] = updated
    if not dry_run:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite absolute image/mask/topology paths in manifest CSVs after moving data."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Manifest CSV files or directories to scan recursively for CSV manifests.",
    )
    parser.add_argument(
        "--old-root",
        required=True,
        type=Path,
        help="Old project root prefix recorded in the manifests, e.g. /Users/.../TopoAnchor.",
    )
    parser.add_argument(
        "--new-root",
        required=True,
        type=Path,
        help="New project root prefix on the training machine, e.g. /workspace/topo-anchor.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would change only.")
    parser.add_argument(
        "--topology-cache-dir",
        type=Path,
        default=None,
        help="Optional topology cache JSON file or directory whose mask_path fields should also be rebased.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_root = args.old_root.expanduser().resolve()
    new_root = args.new_root.expanduser().resolve()
    manifests = _iter_manifest_paths([path.expanduser() for path in args.paths])
    changed_count = 0
    for manifest in manifests:
        changed = rebase_manifest(
            manifest,
            old_root=old_root,
            new_root=new_root,
            dry_run=bool(args.dry_run),
        )
        status = "would update" if args.dry_run and changed else "updated" if changed else "unchanged"
        print(f"{status}: {manifest}")
        changed_count += int(changed)
    print(f"Processed {len(manifests)} CSV files; {changed_count} changed.")
    if args.topology_cache_dir is not None:
        json_paths = _iter_json_paths(args.topology_cache_dir.expanduser())
        json_changed_count = 0
        for json_path in json_paths:
            changed = rebase_topology_cache_json(
                json_path,
                old_root=old_root,
                new_root=new_root,
                dry_run=bool(args.dry_run),
            )
            status = "would update" if args.dry_run and changed else "updated" if changed else "unchanged"
            print(f"{status}: {json_path}")
            json_changed_count += int(changed)
        print(f"Processed {len(json_paths)} topology JSON files; {json_changed_count} changed.")


if __name__ == "__main__":
    main()
