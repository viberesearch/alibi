# Datasette Record Explorer

A read-only web UI for **listing and analysing** Alibi records (facts, line
items, and the materialised `item_stars` analytics surface) without writing
code. It runs on the host as a launchd service, served by
[Datasette](https://datasette.io/) over a periodic snapshot of the live
database.

## Why a snapshot, not the live DB

The API (`lt serve`) owns `data/alibi.db` as a single WAL writer. Datasette
reads a **snapshot** produced by `sqlite3 .backup` (an online, WAL-safe copy),
so the explorer never contends with that writer and always opens a complete,
consistent database. The snapshot is regenerated periodically (see *Refresh*),
so the explorer trails the live data by at most one refresh interval.

```
[ live alibi.db ]  --sqlite3 .backup-->  [ data/explore/alibi-explore.db ]
   (API, WAL writer)   (every 15 min)         (Datasette --immutable, :8001)
```

## Access

Bound to **127.0.0.1:8001** by default. Datasette has no authentication and
this is the full financial database, so it is **not** exposed on the LAN.

- On the host: open <http://127.0.0.1:8001>.
- From another machine: SSH-tunnel — `ssh -L 8001:localhost:8001 <host>` then
  open <http://localhost:8001> — or reach the host over Tailscale.
- To expose deliberately, set `ALIBI_DATASETTE_HOST=0.0.0.0` **and** put
  authentication in front (e.g. a reverse proxy or a Datasette auth plugin).

## What you can do

- Browse and filter every table (faceted browse, full-text where available).
- Run ad-hoc SQL (read-only) and export results as CSV/JSON.
- Use the built-in **canned queries** (defined in `datasette/metadata.yaml`).
  All spend / price queries are normalised to **EUR** via each fact's historical
  rate (run `lt fx backfill`; a fact with no rate yet is excluded, not blended):
  - **Spend by vendor (EUR)** / **Spend by month (EUR)** / **Spend by category (EUR)**
  - **Comparable unit price — cross-vendor (EUR)**: each comparable product's
    EUR-normalised price per standard unit (EUR/kg, EUR/L, EUR/pcs) across vendors.
  - **Cheapest vendor per comparable product (EUR)**: the lowest EUR-normalised
    comparable unit price for each product.
  - **Comparable unit price by product STATE**: each comparable product's
    normalised price broken down by preservation/preparation state (fresh /
    canned / frozen / dried / cured / pickled / roasted / cooked — the state
    facet), limited to products seen in more than one state (real comparisons:
    fresh vs canned vs frozen artichokes; raw vs roasted nuts).
  - **Line items by product STATE**: every priced line item with its `state`
    surfaced from the attributes JSON, for row-level browsing.

`item_stars` is the recommended starting table — one row per line item,
denormalised with vendor, category, `comparable_name`, and
`comparable_unit_price` (the unit of analysis).

## Operations

Files (in the repo):

| Path | Purpose |
|------|---------|
| `scripts/datasette_snapshot.sh` | Produce a WAL-safe snapshot → `data/explore/alibi-explore.db` |
| `scripts/run-datasette.sh` | Launch Datasette (`uv tool run`, `--immutable`, 127.0.0.1:8001) |
| `scripts/datasette_refresh.sh` | Re-snapshot + reload the service (used by the timer) |
| `datasette/metadata.yaml` | Titles, table descriptions, canned queries |

Services (launchd agents on the host, in `~/Library/LaunchAgents/`):

| Label | Role |
|-------|------|
| `com.alibi.datasette` | KeepAlive service running `run-datasette.sh` |
| `com.alibi.datasette-refresh` | Timer (`StartInterval` 900s) → `datasette_refresh.sh` |

Because the service runs with `--immutable` (which pins the snapshot file open),
the refresh job re-snapshots **and** `launchctl kickstart -k`s the service so it
picks up the new data.

### Manage

```bash
# Run locally without launchd (foreground):
./scripts/run-datasette.sh

# Refresh the snapshot now and reload the service:
./scripts/datasette_refresh.sh

# launchd control:
launchctl bootstrap   gui/$(id -u) ~/Library/LaunchAgents/com.alibi.datasette.plist
launchctl bootout     gui/$(id -u)/com.alibi.datasette
launchctl kickstart -k gui/$(id -u)/com.alibi.datasette
tail -f ~/Library/Logs/alibi-datasette.log
```

### Install (one-time)

Datasette runs via `uv tool run --with datasette-vega datasette` (no project
dependency coupling). The first launch downloads it. The launchd plists are
host config (see `~/Library/LaunchAgents/com.alibi.datasette*.plist`);
keep them under version control alongside your other host service definitions.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `ALIBI_DATASETTE_HOST` | `127.0.0.1` | Bind address (set `0.0.0.0` only behind auth) |
| `ALIBI_DATASETTE_PORT` | `8001` | Listen port (any free port) |
| `ALIBI_DB_PATH` | `data/alibi.db` | Source DB for the snapshot |
