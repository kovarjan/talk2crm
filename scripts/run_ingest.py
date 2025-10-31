#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import gzip

from core.ingestion.ingestor import Ingestor
from core.embedding.embedder import Embedder

def load_client_cfg(cfg_path: str, client_name: str) -> dict:
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        clients = json.load(f)
    if not isinstance(clients, list):
        raise ValueError("clients_config.json must be a JSON array")
    for c in clients:
        if c.get("name") == client_name:
            return c
    raise KeyError(f"Client '{client_name}' not found in {cfg_path}")


def new_snapshot_paths(lake_dir: str, tenant: str, module: str, started_at: float) -> list[str]:
    """Return paths of ndjson.gz snapshots created since `started_at`."""
    base = os.path.join(lake_dir, tenant, module)
    if not os.path.isdir(base):
        return []
    out = []
    for fn in os.listdir(base):
        if not fn.endswith(".ndjson.gz"):
            continue
        p = os.path.join(base, fn)
        try:
            if os.path.getmtime(p) >= started_at:
                out.append(p)
        except OSError:
            pass
    return sorted(out)


def stream_items_from_snapshot(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
        Main entry point for running CRM data ingestion and embedding for a selected client.
        This function performs the following steps:
        1. Parses command-line arguments to determine the client, configuration file, directories, modules, and page size.
        2. Loads the client-specific configuration from a JSON file, including per-module field settings.
        3. Optionally filters modules to process based on user input.
        4. Runs the ingestion process for each selected module, pulling data from export endpoints, writing raw snapshots, and updating watermarks.
        5. For each module, builds embeddings for records added during this run by reading new snapshot files and upserting batches of vectors.
        6. Prints progress and summary information to the console.
        Command-line Arguments:
        --client      (required) Name of the client as defined in clients_config.json (e.g., "ai").
        --config      Path to the clients_config.json file (default: "core/clients/clients_config.json").
        --lake-dir    Directory to store raw snapshot outputs (default: "lake").
        --state-dir   Directory to store watermark/state information (default: "state").
        --modules     List of specific modules to process (e.g., "accounts contacts"). If omitted, all modules are processed.
        --page-size   Page size for data export requests (default: 500).
        Returns:
        int: Exit code (0 for success, 2 for configuration or filtering errors).
        Raises:
        SystemExit: If required arguments are missing or configuration is invalid.
        Example usage:
        python run_ingest.py --client ai --modules accounts contacts
        """)
        argparse.ArgumentParser(description="Run CRM ingest + embeddings for a selected client.").print_help()
        sys.exit(0)
    ap = argparse.ArgumentParser(description="Run CRM ingest + embeddings for a selected client.")
    ap.add_argument("--client", required=True, help="Client name from clients_config.json (e.g., ai)")
    ap.add_argument("--config", default="core/clients/clients_config.json", help="Path to clients_config.json")
    ap.add_argument("--lake-dir", default="var/lake", help="Raw snapshot output directory")
    ap.add_argument("--state-dir", default="var/state", help="Watermark/state directory")
    ap.add_argument("--modules", nargs="*", help="Limit to specific modules (e.g., accounts contacts)")
    ap.add_argument("--page-size", type=int, default=500)

    args = ap.parse_args()

    started_at = time.time()

    # Load per-module field config from clients_config.json
    client_cfg = load_client_cfg(args.config, args.client)
    modules_cfg = client_cfg.get("modules") or {}
    if not modules_cfg:
        print(f"No 'modules' config for client '{args.client}'", file=sys.stderr)
        return 2

    # Filter modules if requested from cli
    modules = list(modules_cfg.keys())
    if args.modules:
        requested = set(args.modules)
        modules = [m for m in modules if m in requested]
        if not modules:
            print("No matching modules after filtering.", file=sys.stderr)
            return 2

    # Run ingest (pull from /export endpoints, write snapshots, update watermarks)
    ing = Ingestor(args.config, state_dir=args.state_dir, out_dir=args.lake_dir)
    print(f"[ingest] Starting ingest for client '{args.client}' modules: {modules}")
    for module in modules:
        fields = modules_cfg[module].get("fields", [])
        include_rel = True  # keep relationships enabled; they help with embeddings
        print(f"[ingest] {args.client}/{module} fields={fields}")
        # NOTE: Ingestor.run_module handles pagination + watermark and writes snapshots
        ing.run_module(args.client, module, fields, include_rel=include_rel)

    # Build embeddings for records added in this run (read new snapshot files)
    emb = Embedder()  # uses your default model/store
    for module in modules:
        snapshots = new_snapshot_paths(args.lake_dir, args.client, module, started_at)
        if not snapshots:
            print(f"[embed] {args.client}/{module}: no new snapshots.")
            continue

        batch = []
        count = 0
        fields = modules_cfg[module].get("fields", [])

        for snap in snapshots:
            for rec in stream_items_from_snapshot(snap):
                batch.append(rec)
                if len(batch) >= 512:  # tune batch size for your GPU/CPU
                    emb.upsert_batch(args.client, module, batch, fields)
                    count += len(batch)
                    batch.clear()
        if batch:
            emb.upsert_batch(args.client, module, batch, fields)
            count += len(batch)

        print(f"[embed] {args.client}/{module}: upserted {count} vectors from {len(snapshots)} snapshot(s).")

    print("[done] ingest+embed complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
