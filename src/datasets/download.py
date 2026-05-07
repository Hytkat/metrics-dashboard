# pyright: reportAny=false, reportUnknownVariableType=false

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import cast

import kagglehub

CACHE_DIR = Path("cache")

RegistryDataset = dict[str, str]
Registry = dict[str, dict[str, RegistryDataset]]


def load_registry() -> Registry:
    registry_file = Path(__file__).parent / "datasets.json"
    if registry_file.exists():
        return cast(Registry, json.loads(registry_file.read_text()))
    return {"datasets": {}}


def save_registry(registry: Registry) -> None:
    registry_file = Path(__file__).parent / "datasets.json"
    _ = registry_file.write_text(json.dumps(registry, indent=2) + "\n")


def download(dataset_key: str, force: bool = False) -> None:
    registry = load_registry()
    if dataset_key not in registry["datasets"]:
        print(
            f"Error: '{dataset_key}' not found in registry. Use 'list' to see available datasets."
        )
        sys.exit(1)

    info: RegistryDataset = registry["datasets"][dataset_key]
    handle: str = info["handle"]
    dest = CACHE_DIR.resolve() / dataset_key

    if dest.exists() and not force:
        print(
            f"Dataset '{dataset_key}' already exists at {dest}. Use --force to redownload."
        )
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{dataset_key}' (handle: {handle})...")
    os.environ["KAGGLEHUB_CACHE"] = str(CACHE_DIR.resolve())
    _kaggle_source = kagglehub.dataset_download(handle)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    kaggle_source = Path(cast(str, _kaggle_source))
    if kaggle_source.is_dir():
        for item in kaggle_source.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    _ = shutil.rmtree(target)
                else:
                    _ = target.unlink()
            if item.is_dir():
                _ = shutil.copytree(item, target)
            else:
                _ = shutil.copy2(item, target)
    kaggle_datasets = CACHE_DIR.resolve() / "datasets"
    if kaggle_datasets.exists():
        _ = shutil.rmtree(kaggle_datasets)
    print(f"Dataset '{dataset_key}' downloaded to {dest}")


def download_all(force: bool = False) -> None:
    registry = load_registry()
    if not registry["datasets"]:
        print("No datasets registered. Use 'add' to register a dataset.")
        return

    for key in registry["datasets"]:
        try:
            download(key, force=force)
        except Exception as e:
            print(f"Error downloading '{key}': {e}")


def add(dataset_key: str, handle: str) -> None:
    registry = load_registry()
    if dataset_key in registry["datasets"]:
        print(f"Error: '{dataset_key}' already exists in registry.")
        sys.exit(1)

    registry["datasets"][dataset_key] = {"handle": handle}
    save_registry(registry)
    print(f"Added '{dataset_key}' to registry with handle: {handle}")


def list_datasets() -> None:
    registry = load_registry()
    if not registry["datasets"]:
        print("No datasets registered. Use 'add' to register a dataset.")
        return

    print("Registered datasets:")
    for key, info in registry["datasets"].items():
        dest = CACHE_DIR / key
        status = "downloaded" if dest.exists() else "not downloaded"
        print(f"  [{status}] {key}: {info['handle']}")


class ParsedArgs(argparse.Namespace):
    command: str = ""
    dataset: str | None = None
    all: bool = False
    force: bool = False
    handle: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and manage Kaggle datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dl_parser = subparsers.add_parser("download", help="Download a registered dataset")
    _ = dl_parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset key from registry (omit for --all)",
    )
    _ = dl_parser.add_argument(
        "--all", action="store_true", help="Download all registered datasets"
    )
    _ = dl_parser.add_argument("--force", action="store_true", help="Force redownload")

    add_parser = subparsers.add_parser("add", help="Add a dataset to the registry")
    _ = add_parser.add_argument("dataset", help="Dataset key name")
    _ = add_parser.add_argument(
        "handle", help="Kaggle dataset handle (e.g., owner/dataset-name)"
    )

    _ = subparsers.add_parser("list", help="List all registered datasets")

    args = parser.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)

    if args.command == "download":
        if args.all or args.dataset is None:
            download_all(force=args.force)
        else:
            download(args.dataset, force=args.force)
    elif args.command == "add":
        assert args.dataset is not None
        add(args.dataset, args.handle)
    elif args.command == "list":
        list_datasets()
