"""Analytics stack subscriber — pushes facts on create/update events."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from alibi.db.connection import DatabaseManager
from alibi.services.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class AnalyticsExportSubscriber:
    """Pushes a full export to the analytics stack on fact events.

    Subscribes to FACT_CREATED and FACT_UPDATED. Each event triggers a
    full-replace export in a background daemon thread so the main
    pipeline is never blocked.

    A periodic timer thread also runs as a safety net, exporting on a
    fixed interval even when no events are fired.
    """

    def __init__(
        self,
        db_factory: Callable[[], DatabaseManager],
        analytics_url: str,
        bus: EventBus | None = None,
        timeout: float = 30.0,
        export_interval: float = 300.0,
    ) -> None:
        self._db_factory = db_factory
        self._analytics_url = analytics_url
        self._bus = bus
        self._timeout = timeout
        self._export_interval = export_interval
        self._shutdown_event = threading.Event()
        self._timer_thread: threading.Thread | None = None
        self.export_count = 0

    def start(self) -> None:
        """Subscribe to fact events on the bus and start the periodic timer."""
        if self._bus is None:
            from alibi.services.events import event_bus

            self._bus = event_bus
        self._bus.subscribe(EventType.FACT_CREATED, self._handle)
        self._bus.subscribe(EventType.FACT_UPDATED, self._handle)

        self._shutdown_event.clear()
        self._timer_thread = threading.Thread(
            target=self._timer_loop,
            daemon=True,
            name="analytics-export-timer",
        )
        self._timer_thread.start()

    def stop(self) -> None:
        """Unsubscribe from all events and shut down the periodic timer."""
        if self._bus is None:
            return
        self._bus.unsubscribe(EventType.FACT_CREATED, self._handle)
        self._bus.unsubscribe(EventType.FACT_UPDATED, self._handle)

        self._shutdown_event.set()
        if self._timer_thread is not None:
            self._timer_thread.join(timeout=5.0)
            self._timer_thread = None

    def _timer_loop(self) -> None:
        """Periodic export fallback — runs until shutdown."""
        while not self._shutdown_event.is_set():
            self._shutdown_event.wait(timeout=self._export_interval)
            if self._shutdown_event.is_set():
                break
            try:
                db = self._db_factory()
                from alibi.services.export_analytics import push_to_analytics_stack

                push_to_analytics_stack(db, self._analytics_url, timeout=self._timeout)
                self.export_count += 1
                logger.info("Scheduled analytics export complete")
            except ConnectionError:
                logger.warning(
                    "Scheduled analytics export failed (endpoint unreachable)"
                )
            except Exception:
                logger.exception("Unexpected error during scheduled analytics export")

    def _handle(self, event: Event) -> None:
        """Handle a fact event by triggering export in a background thread."""
        thread = threading.Thread(
            target=self._export,
            args=(event,),
            daemon=True,
        )
        thread.start()

    def _export(self, event: Event) -> None:
        """Run the full export to the analytics stack."""
        from alibi.services.export_analytics import push_to_analytics_stack

        try:
            db = self._db_factory()
            result = push_to_analytics_stack(
                db, self._analytics_url, timeout=self._timeout
            )
            self.export_count += 1
            logger.info(
                "Analytics export triggered by %s: %d facts",
                event.type.value,
                result["facts_count"],
            )
        except ConnectionError:
            logger.warning(
                "Analytics export failed for %s (endpoint unreachable)",
                event.type.value,
            )
        except Exception:
            logger.exception(
                "Unexpected error during analytics export for %s",
                event.type.value,
            )
