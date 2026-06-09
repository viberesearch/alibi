# data/

Runtime data directory. All contents are gitignored except example files.

| Path | Purpose | Gitignored |
|------|---------|------------|
| `alibi.db` | SQLite database | Yes |
| `yaml_store/` | YAML extraction cache (`.alibi.yaml` files) | Yes |
| `uploads/` | Uploaded document files | Yes |
| `lancedb/` | LanceDB vector store (optional) | Yes |
| `vendor_aliases.yaml` | Your vendor name mappings | Yes |
| `unit_aliases.yaml` | Your unit type mappings | Yes |
| `vendor_aliases.example.yaml` | Example vendor aliases (copy and customize) | No |
| `unit_aliases.example.yaml` | Example unit aliases (copy and customize) | No |

## First-time Setup

```bash
cp data/vendor_aliases.example.yaml data/vendor_aliases.yaml
cp data/unit_aliases.example.yaml data/unit_aliases.yaml
# Edit both files to match your local vendors and units
```
