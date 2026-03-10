"""Mycelium integration for Obsidian vault processing.

This module provides integration between Alibi and the Mycelium
Obsidian vault architecture, enabling:

- Watching the vault inbox for new documents
- Processing documents from iOS sync (via Working Copy)
- Generating Obsidian notes for processed artifacts
- Git-aware sync detection for batch processing
"""

from alibi.mycelium.watcher import MyceliumWatcher
from alibi.mycelium.notes import generate_artifact_note, ArtifactNoteExporter
from alibi.mycelium.sync import SyncDetector, process_after_sync

__all__ = [
    "MyceliumWatcher",
    "generate_artifact_note",
    "ArtifactNoteExporter",
    "SyncDetector",
    "process_after_sync",
]
