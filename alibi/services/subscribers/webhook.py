"""Webhook dispatcher — delivers events to external systems via HTTP POST."""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import threading
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from alibi.services.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


def _is_private_url(url: str) -> bool:
    """Check if a URL resolves to a private/reserved IP range (SSRF protection)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return True
        # Resolve hostname to IP addresses
        for info in socket.getaddrinfo(hostname, parsed.port or 80):
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return True
    except (socket.gaierror, ValueError):
        return True
    return False


class WebhookDispatcher:
    """Dispatches events to an HTTP endpoint as JSON POST requests.

    Non-blocking: uses a daemon thread for each delivery so the main
    pipeline is never blocked by webhook latency.
    Validates URLs against private IP ranges to prevent SSRF.
    """

    def __init__(
        self,
        url: str,
        bus: EventBus | None = None,
        timeout: float = 5.0,
        allow_private: bool = False,
    ) -> None:
        if not allow_private and _is_private_url(url):
            raise ValueError(
                f"Webhook URL resolves to a private/reserved IP: {url}. "
                "Set allow_private=True to override."
            )
        self._url = url
        self._timeout = timeout
        self._bus = bus

    def start(self) -> None:
        """Subscribe to all event types on the bus."""
        if self._bus is None:
            from alibi.services.events import event_bus

            self._bus = event_bus
        for event_type in EventType:
            self._bus.subscribe(event_type, self._handle)

    def stop(self) -> None:
        """Unsubscribe from all event types."""
        if self._bus is None:
            return
        for event_type in EventType:
            self._bus.unsubscribe(event_type, self._handle)

    def _handle(self, event: Event) -> None:
        """Handle an event by dispatching in a background thread."""
        thread = threading.Thread(
            target=self._deliver,
            args=(event,),
            daemon=True,
        )
        thread.start()

    def _deliver(self, event: Event) -> None:
        """POST the event payload to the configured URL."""
        payload = {
            "type": event.type.value,
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
        }
        try:
            body = json.dumps(payload).encode("utf-8")
            req = Request(
                self._url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=self._timeout) as resp:
                logger.debug(
                    "Webhook delivered %s -> %s (%d)",
                    event.type.value,
                    self._url,
                    resp.status,
                )
        except (URLError, OSError) as exc:
            logger.warning(
                "Webhook delivery failed for %s -> %s: %s",
                event.type.value,
                self._url,
                exc,
            )
        except Exception:
            logger.exception(
                "Unexpected error delivering webhook for %s",
                event.type.value,
            )
