# Bulk Data Corrections

Edit fact_items directly in any SQLite GUI tool (TablePlus, DB Browser, DBeaver) and sync changes back into alibi's correction and learning system.

## Why This Workflow Exists

Direct SQL edits bypass alibi's correction pipeline. That means:

- No correction events are logged (the adaptive learning system does not see your changes)
- No sibling propagation (fixing a brand on one "Milk" will not fix the others)
- No enrichment_source stamp (the system does not know the value is user-confirmed)

The snapshot workflow bridges this gap. Edit freely in your GUI tool, then a single CLI command captures everything you changed as proper correction events.

## Quick Reference

```bash
uv run lt corrections snapshot          # 1. Save current state
# ... edit in your SQLite GUI tool ...
uv run lt corrections detect            # 2. Preview changes (dry-run)
uv run lt corrections detect --apply    # 3. Record changes + propagate
```

## Step-by-Step

### 1. Connect to the Database

Open the alibi SQLite database in your preferred tool:

- **Default path**: `data/alibi.db` (relative to project root)
- If `ALIBI_DB_PATH` is set in your `.env`, use that path instead

### 2. Take a Snapshot

Before editing, save the current state of all fact_items:

```bash
uv run lt corrections snapshot
```

This stores a copy of every item's brand, category, comparable_name, unit, unit_quantity, and barcode.

### 3. Edit in Your GUI Tool

Open the `fact_items` table and edit fields directly. Common corrections:

| Field | What to Edit | Example |
|-------|-------------|---------|
| `brand` | Product manufacturer | "" -> "Barilla" |
| `category` | Spending category | "uncategorized" -> "Pasta" |
| `comparable_name` | Standardized English name | "" -> "penne rigate" |
| `unit` | Unit type (kg, l, pcs, pack) | "pcs" -> "kg" |
| `unit_quantity` | Package size for comparison | NULL -> 0.5 |

### 4. Preview Changes

See what you changed without recording anything:

```bash
uv run lt corrections detect
```

This shows a diff of every field that changed since the snapshot.

### 5. Apply Changes

Record all changes as correction events and trigger learning propagation:

```bash
uv run lt corrections detect --apply
```

This will:
- Create a correction event for every changed field
- Stamp the enrichment_source as `user_confirmed` (confidence 1.0)
- Propagate brand/category changes to matching items via sibling propagation
- Update the vendor's correction rate statistics
