"""
SDRA V4.0 - Autonomous Core Engine
==================================
The 'AppEngine' is a Singleton that manages the application state and event dispatching.
It replaces the direct coupling between UI (main.py) and Logic (workers, scanners).

Philosophy:
----------
1. UI subscribes to events (Observer Pattern).
2. Engine dispatches events asynchronously.
3. No blocking I/O on the main thread (managed by built-in Executor).
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
import queue

# Re-export key components for easy access via engine
from database import SRDADatabase

logger = logging.getLogger('srda.engine')

class AppEvent(Enum):
    """System-wide Events."""
    DOC_INGESTED = "DOC_INGESTED"     # File added to DB
    DOC_PROCESSED = "DOC_PROCESSED"   # Extraction complete
    DOC_MATCHED = "DOC_MATCHED"       # Reconciliation Suggestion found
    DOC_UPDATED = "DOC_UPDATED"       # Manual edit saved
    STATUS_CHANGED = "STATUS_CHANGED" # General status update (for StatusBar)
    ERROR_OCCURRED = "ERROR_OCCURRED" # Global error handler

@dataclass
class EventData:
    """Payload for events."""
    event_type: AppEvent
    data: Any # Dict, DocumentNode, or Message String
    source: str = "SYSTEM"

class AppEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AppEngine, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        
        self._initialized = True
        self.db = SRDADatabase()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="SdraWorker")
        self._subscribers: Dict[AppEvent, List[Callable[[EventData], None]]] = {}
        
        # UI Thread Queue (UI polls this to update safely)
        self.ui_queue = queue.Queue()
        
        logger.info("🚀 AppEngine V4.0 (Autonomous Core) Initialized")

    def subscribe(self, event_type: AppEvent, callback: Callable[[EventData], None]):
        """Register a callback for an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.name}")

    def dispatch(self, event_type: AppEvent, data: Any, source: str = "SYSTEM"):
        """
        Dispatch an event to all subscribers.
        NOTE: This might run on a background thread!
        UI subscribers must handle thread safety (or use the ui_queue mechanism).
        """
        event = EventData(event_type, data, source)
        
        # 1. Direct Subscribers (careful with threads)
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
        
        # 2. Global Logging
        if event_type == AppEvent.ERROR_OCCURRED:
            logger.error(f"Event Error: {data}")
        elif event_type == AppEvent.STATUS_CHANGED:
            logger.info(f"Status: {data}")

    def run_task(self, func: Callable, *args, **kwargs):
        """Submit a task to the background thread pool."""
        self.executor.submit(self._task_wrapper, func, *args, **kwargs)

    def _task_wrapper(self, func, *args, **kwargs):
        """Wraps tasks to catch errors and dispatch ERROR_OCCURRED."""
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Task failed: {func.__name__}")
            self.dispatch(AppEvent.ERROR_OCCURRED, {"message": str(e), "context": func.__name__})

    def shutdown(self):
        """Cleanup resources."""
        logger.info("Shutting down AppEngine...")
        self.executor.shutdown(wait=False)
        self.db.close()

# Global Accessor
def get_engine() -> AppEngine:
    return AppEngine()
