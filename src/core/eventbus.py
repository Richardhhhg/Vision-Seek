import asyncio
import logging
from typing import Any

logger = logging.getLogger("core")


class EventBus:
    """
    Eventbus communication module for the different services. Each service has a queue which then
    subscribes to different events. When an event is published, the eventbus puts that event into
    the queue of each service subscribed to that event type.
    """

    def __init__(self):
        self._subscribers: dict[Any, list[asyncio.Queue]] = {}

    def subscribe(self, event_type: Any, queue: asyncio.Queue):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        logger.debug(f"Subscribing new queue to event {event_type}")
        self._subscribers[event_type].append(queue)

    def unsubscribe(self, event_type: Any, queue: asyncio.Queue):
        if event_type in self._subscribers:
            logger.debug(f"Unsubscribing queue from event {event_type}")
            self._subscribers[event_type].remove(queue)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]
        else:
            logger.warning(
                f"Attempted to unsubscribe from event type {event_type} which has no subscribers"
            )

    async def publish(self, event: Any) -> None:
        event_type = type(event)
        if event_type in self._subscribers:
            logger.debug(
                f"Publishing event {event_type} to {len(self._subscribers[event_type])} subscribers"
            )
            for queue in self._subscribers[event_type]:
                await queue.put(event)
        else:
            logger.warning(f"No subscribers for event type: {event_type}")
