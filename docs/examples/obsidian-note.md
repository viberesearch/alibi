# Example: Obsidian Note Output

When alibi processes a receipt and the Obsidian notes subscriber is enabled, it generates a markdown note like this:

```markdown
---
type: transaction
id: "2b153b6e-78b2-46e5-af23-081e906ac087"
date: 2026-03-01
vendor: "Alphamega"
amount: 45.67
currency: EUR
category: "groceries"
tags: []
status: confirmed
---

# Alphamega - EUR 45.67

| | |
|---|---|
| **Date** | 2026-03-01 |
| **Time** | 12:30:47 |
| **Amount** | EUR 45.67 |
| **Payment** | card |
| **Category** | groceries |
| **Status** | confirmed |

## Vendor

**VAT**: 10XXXXXXY

## Line Items

| Item | Brand | Qty | Unit | Size | Unit Price | Total | EUR/std | Category |
|------|-------|-----|------|------|-----------|-------|---------|----------|
| Organic Milk 1L | Alphamega | 2 | pcs | 1L | 2.35 | 4.70 EUR | 2.35/l | Dairy |
| Barilla Penne Rigate 500g | Barilla | 1 | g | 500g | 1.95 | 1.95 EUR | 3.90/kg | Pasta |
| Avocado | | 0.85 | kg | 1kg | 3.59 | 3.05 EUR | 3.59/kg | Produce |
| Greek Yogurt 200g | Charalambides | 3 | g | 200g | 1.49 | 4.47 EUR | 7.45/kg | Dairy |

## Linked Documents

- [[receipt.jpeg]] (receipt)
```

The `EUR/std` column shows the standardized per-unit price (e.g., EUR per kg or EUR per liter), calculated from `unit_quantity` and `unit`. This enables price comparison across vendors and package sizes.
