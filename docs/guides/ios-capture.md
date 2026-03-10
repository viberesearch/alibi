# iOS Shortcuts for Receipt Capture

> **Requirements**: iOS 16+, Obsidian app, optionally Working Copy (git client)

## Overview

Three iOS Shortcuts to capture receipts and expenses directly from your phone into alibi's inbox:

1. **Receipt Capture** -- Photo to timestamped document
2. **Quick Expense Log** -- Markdown expense note
3. **Vault Sync** -- Push changes to your server (Working Copy only)

All shortcuts save to your vault's `inbox/` directory via iCloud Drive or Working Copy.

## Prerequisites

- **Obsidian** on iOS with your vault configured
- Vault folder structure:
  ```
  your-vault/
  └── inbox/
      ├── documents/     # Receipt images go here
      └── expenses/      # Expense notes go here
  ```

### Sync Options

| | iCloud Drive | Working Copy |
|---|---|---|
| Setup | Simple (2 min) | Medium (5 min) |
| Sync | Automatic | Manual or scheduled |
| Git history | No | Yes |
| Best for | Most users | Developers |

## Shortcut 1: Receipt Capture

Captures a photo and saves it with a structured filename.

### Flow

```
[Ask for Photo] -> [Get Date] -> [Ask Vendor] -> [Ask Type]
    -> [Save as: YYYY-MM-DD_Vendor_Type.jpg to inbox/documents/]
```

### Setup

1. Open **Shortcuts** app, tap **+**
2. Add actions in order:

| Step | Action | Config |
|------|--------|--------|
| 1 | Ask for Photo | Prompt: "Select receipt photo" |
| 2 | Get Current Date | Variable: `current_date` |
| 3 | Format Date | Format: "YYYY-MM-dd", variable: `formatted_date` |
| 4 | Ask for Text | Prompt: "Vendor name?", variable: `vendor_name` |
| 5 | Ask for Menu | Options: Receipt/Invoice/Statement/Warranty, variable: `doc_type` |
| 6 | Text | `[formatted_date]_[vendor_name]_[doc_type]`, variable: `filename` |
| 7 | Save File | Name: `[filename].jpg`, destination: `your-vault/inbox/documents/` |
| 8 | Show Result | "Saved as [filename].jpg" |

### Example

1. Tap shortcut
2. Take photo of receipt
3. Enter "Alphamega"
4. Select "Receipt"
5. Saved as: `2026-03-01_Alphamega_Receipt.jpg`
6. Alibi's file watcher detects and processes automatically

## Shortcut 2: Quick Expense Log

Creates a markdown expense entry without a receipt photo.

### Flow

```
[Ask Amount] -> [Ask Vendor] -> [Ask Category]
    -> [Create Markdown] -> [Save to inbox/expenses/]
```

### Generated File

`inbox/expenses/2026-03-01-Alphamega-expense.md`:

```markdown
---
date: 2026-03-01 14:35
amount: 45.23
vendor: Alphamega
category: category/groceries
status: pending
---

# Expense: Alphamega

- Amount: EUR 45.23
- Vendor: Alphamega
- Category: category/groceries
- Date: 2026-03-01 14:35

## Notes


```

### Suggested Categories

- `category/groceries` -- Supermarkets, food shopping
- `category/dining` -- Restaurants, cafes
- `category/utilities` -- Gas, water, electricity
- `category/shopping` -- Clothing, general goods
- `category/transport` -- Fuel, public transport, parking
- `category/health` -- Doctor, pharmacy
- `category/entertainment` -- Movies, subscriptions

## Shortcut 3: Vault Sync

### iCloud Drive

Sync is automatic. No shortcut needed.

### Working Copy

```
Shortcuts -> Open URL: working-copy://push/your-vault?key=YOUR_REPO_KEY
```

Get your repo key from Working Copy app menu.

## Tips

- **Take photos immediately** after purchase (accurate date, better lighting)
- **Include all text** in the frame -- vendor name, items, total
- **Avoid glare** -- tilt the receipt slightly
- **Use consistent vendor names** -- "Alphamega" not "alphamega" or "ALPHAMEGA"
- Alibi normalizes names automatically, but consistency helps matching
