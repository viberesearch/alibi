# Project Decision Traces

This folder captures decision traces for this project.

## Structure

```
.claude-decisions/
├── README.md           # This file
├── .gitignore          # Excludes raw traces from git
├── traces/             # Raw decision trace JSON files (not committed)
├── local-patterns.md   # Patterns specific to this project (committed)
└── pending-review/     # Traces awaiting promotion review (not committed)
```

## Usage

### Automatic Capture
Traces are automatically captured by Claude Code Engine during development sessions.

### Manual Capture
Use the `/trace` command to manually log important decisions:

```bash
/trace pattern "Description of a reusable pattern"
/trace exception "Rule X was broken because Y"
/trace lesson "Unexpected outcome taught Z"
/trace correction "Human corrected approach from A to B"
```

### View Traces
```bash
/show-traces           # List recent traces for this project
/show-traces --all     # Include archived traces
```

## Local Patterns

Document project-specific patterns in `local-patterns.md`. These are patterns that:
- Only apply to this project
- Are not general enough for fleet-wide promotion
- Reference project-specific architecture or conventions

## Promotion

Run `/review-traces` to review accumulated traces and promote valuable ones to global patterns in `fleet-standards/decision-graph/`.

### Promotion Criteria

A trace should be promoted if it:
1. Applies to multiple projects (not just this one)
2. Represents a reusable insight or approach
3. Documents an important precedent for future decisions

## Storage Policy

| Content | Committed to Git |
|---------|-----------------|
| `README.md` | Yes |
| `local-patterns.md` | Yes |
| `traces/*.json` | No (gitignored) |
| `pending-review/*.json` | No (gitignored) |

Raw traces are stored locally only. Promoted patterns become permanent documentation.
