---
type: dashboard
title: Life Tracker Dashboard
---

# Life Tracker Dashboard

## Recent Transactions

```dataview
TABLE date, amount, vendor, status
FROM "finances/transactions"
WHERE type = "transaction"
SORT date DESC
LIMIT 10
```

## Pending Documents

```dataview
TABLE file.ctime as "Added", type
FROM "inbox/documents"
WHERE !completed
SORT file.ctime DESC
```

## Expiring Warranties

```dataview
TABLE name, warranty_expires, category
FROM "finances/items"
WHERE type = "item" AND warranty_expires != null
WHERE date(warranty_expires) < date(today) + dur(90 days)
SORT warranty_expires ASC
```

## Monthly Spending

```dataview
TABLE sum(number(amount)) as "Total", count(file.name) as "Count"
FROM "finances/transactions"
WHERE type = "transaction" AND date >= date(today) - dur(30 days)
GROUP BY category
SORT Total DESC
```

## Items by Category

```dataview
TABLE count(file.name) as "Count", sum(number(purchase_price)) as "Total Value"
FROM "finances/items"
WHERE type = "item"
GROUP BY category
SORT Total Value DESC
```

## Insurance Inventory

```dataview
TABLE name, category, purchase_price, warranty_expires
FROM "finances/items"
WHERE type = "item" AND insurance_covered = true
SORT category ASC
```
