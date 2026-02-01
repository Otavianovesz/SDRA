"""
SDRA Workers - Professional Threading System
=============================================

This module implements a robust async worker system for SDRA.
All heavy operations run in background threads with proper signal/callback
mechanisms to update the UI without blocking.

Architecture:
    GUI (Main Thread) <--queue--> WorkerPool (Background Threads)
    
    - TaskQueue: Thread-safe queue for pending work
    - WorkerPool: Manages ThreadPoolExecutor  
    - TaskResult: Dataclass for worker results
    - Callbacks: UI updates via root.after()

Usage:
    pool = WorkerPool(max_workers=4)
    pool.submit(
        task_type="extract",
        task_data={"file": path},
        on_progress=lambda p, t, m: update_ui(p, t, m),
        on_complete=lambda r: show_result(r),
        on_error=lambda e: show_error(e)
    )
"""

import logging
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum, auto

logger = logging.getLogger('srda.workers')


class TaskStatus(Enum):
    """Status of a task in the worker system."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class TaskType(Enum):
    """Types of tasks that can be processed."""
    EXTRACT_FILE = "extract_file"
    EXTRACT_BATCH = "extract_batch"
    GEMINI_PROCESS = "gemini_process"
    GEMINI_BATCH = "gemini_batch"
    RECONCILE = "reconcile"
    GMAIL_SYNC = "gmail_sync"
    RENAME_FILES = "rename_files"
    IMPORT_FILES = "import_files"


@dataclass
class TaskResult:
    """Result from a worker task."""
    task_id: str
    task_type: TaskType
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: int = 0
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


@dataclass  
class Task:
    """A task to be executed by a worker."""
    task_id: str
    task_type: TaskType
    data: Dict[str, Any]
    on_progress: Optional[Callable[[int, int, str], None]] = None
    on_complete: Optional[Callable[[TaskResult], None]] = None
    on_error: Optional[Callable[[str], None]] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    
    def __hash__(self):
        return hash(self.task_id)


class ProgressTracker:
    """
    Thread-safe progress tracker for batch operations.
    Aggregates progress from multiple files and provides smooth updates.
    """
    
    def __init__(self, total: int, callback: Callable[[int, int, str], None] = None):
        self.total = total
        self.current = 0
        self.message = ""
        self.callback = callback
        self._lock = threading.Lock()
        self._last_update = 0
        self._update_interval = 0.1  # Min seconds between UI updates
        
    def update(self, current: int = None, message: str = None, force: bool = False):
        """Update progress, throttling UI updates."""
        with self._lock:
            if current is not None:
                self.current = current
            if message is not None:
                self.message = message
                
            now = time.time()
            if force or (now - self._last_update) >= self._update_interval:
                self._last_update = now
                if self.callback:
                    try:
                        self.callback(self.current, self.total, self.message)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
    
    def increment(self, message: str = None):
        """Increment progress by 1."""
        with self._lock:
            self.current += 1
        self.update(message=message)
    
    @property
    def percent(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100


class WorkerPool:
    """
    Manages a pool of worker threads for async task execution.
    
    Features:
    - ThreadPoolExecutor for efficient thread reuse
    - Task queue with priority support
    - Progress callbacks for UI updates
    - Graceful shutdown
    - Task cancellation
    
    Example:
        pool = WorkerPool(max_workers=4)
        
        def on_progress(current, total, msg):
            root.after(0, lambda: status_bar.config(text=f"{current}/{total}: {msg}"))
        
        def on_complete(result):
            root.after(0, lambda: refresh_table())
        
        pool.submit(
            task_type=TaskType.EXTRACT_FILE,
            data={"file_path": "/path/to/file.pdf"},
            on_progress=on_progress,
            on_complete=on_complete
        )
    """
    
    def __init__(self, max_workers: int = 4, tk_root = None):
        """
        Initialize worker pool.
        
        Args:
            max_workers: Maximum concurrent workers
            tk_root: Tkinter root for safe UI updates via after()
        """
        self.max_workers = max_workers
        self.tk_root = tk_root
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="srda_worker")
        self._tasks: Dict[str, Task] = {}
        self._futures: Dict[str, Future] = {}
        self._task_counter = 0
        self._lock = threading.Lock()
        self._shutdown = False
        
        # Task handlers - register handlers for each task type
        self._handlers: Dict[TaskType, Callable] = {}
        
        logger.info(f"WorkerPool initialized with {max_workers} workers")
    
    def register_handler(self, task_type: TaskType, handler: Callable):
        """
        Register a handler function for a task type.
        
        Args:
            task_type: The type of task this handler processes
            handler: Callable that takes (task_data, progress_callback) and returns result
        """
        self._handlers[task_type] = handler
        logger.debug(f"Registered handler for {task_type.value}")
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        with self._lock:
            self._task_counter += 1
            return f"task_{self._task_counter}_{int(time.time() * 1000)}"
    
    def submit(
        self,
        task_type: TaskType,
        data: Dict[str, Any],
        on_progress: Callable[[int, int, str], None] = None,
        on_complete: Callable[[TaskResult], None] = None,
        on_error: Callable[[str], None] = None
    ) -> str:
        """
        Submit a task for async execution.
        
        Args:
            task_type: Type of task to execute
            data: Task-specific data dict
            on_progress: Callback(current, total, message) for progress updates
            on_complete: Callback(TaskResult) when task finishes
            on_error: Callback(error_message) if task fails
            
        Returns:
            task_id: Unique identifier for tracking/cancellation
        """
        if self._shutdown:
            raise RuntimeError("WorkerPool is shut down")
        
        task_id = self._generate_task_id()
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            data=data,
            on_progress=on_progress,
            on_complete=on_complete,
            on_error=on_error
        )
        
        with self._lock:
            self._tasks[task_id] = task
        
        # Submit to executor
        future = self._executor.submit(self._execute_task, task)
        self._futures[task_id] = future
        
        # Add completion callback
        future.add_done_callback(lambda f: self._on_task_done(task_id, f))
        
        logger.info(f"Submitted task {task_id} ({task_type.value})")
        return task_id
    
    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a task in worker thread."""
        start_time = time.time()
        task.status = TaskStatus.RUNNING
        
        try:
            handler = self._handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler registered for {task.task_type.value}")
            
            # Create progress wrapper for safe UI updates
            def safe_progress(current: int, total: int, message: str):
                if task.on_progress and self.tk_root:
                    self.tk_root.after(0, lambda: task.on_progress(current, total, message))
                elif task.on_progress:
                    task.on_progress(current, total, message)
            
            # Execute handler
            result_data = handler(task.data, safe_progress)
            
            task.status = TaskStatus.COMPLETED
            return TaskResult(
                task_id=task.task_id,
                task_type=task.task_type,
                success=True,
                data=result_data or {},
                duration_ms=int((time.time() - start_time) * 1000)
            )
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            return TaskResult(
                task_id=task.task_id,
                task_type=task.task_type,
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000)
            )
    
    def _on_task_done(self, task_id: str, future: Future):
        """Handle task completion."""
        task = self._tasks.get(task_id)
        if not task:
            return
        
        try:
            result = future.result()
            
            # Call completion callback on main thread
            if result.success and task.on_complete:
                if self.tk_root:
                    self.tk_root.after(0, lambda: task.on_complete(result))
                else:
                    task.on_complete(result)
                    
            elif not result.success and task.on_error:
                if self.tk_root:
                    self.tk_root.after(0, lambda: task.on_error(result.error))
                else:
                    task.on_error(result.error)
                    
        except Exception as e:
            logger.error(f"Error in task done callback: {e}")
            if task.on_error:
                if self.tk_root:
                    self.tk_root.after(0, lambda: task.on_error(str(e)))
                else:
                    task.on_error(str(e))
        
        finally:
            # Cleanup
            with self._lock:
                self._futures.pop(task_id, None)
    
    def cancel(self, task_id: str) -> bool:
        """
        Attempt to cancel a pending task.
        
        Returns:
            True if task was cancelled, False if already running/completed
        """
        future = self._futures.get(task_id)
        if future and future.cancel():
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.CANCELLED
            logger.info(f"Cancelled task {task_id}")
            return True
        return False
    
    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task."""
        task = self._tasks.get(task_id)
        return task.status if task else None
    
    def get_pending_count(self) -> int:
        """Get count of pending/running tasks."""
        with self._lock:
            return sum(1 for t in self._tasks.values() 
                      if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING))
    
    def shutdown(self, wait: bool = True, timeout: float = 30):
        """
        Shutdown the worker pool.
        
        Args:
            wait: Wait for pending tasks to complete
            timeout: Max seconds to wait
        """
        self._shutdown = True
        logger.info("Shutting down WorkerPool...")
        
        # Cancel pending tasks
        for task_id in list(self._futures.keys()):
            self.cancel(task_id)
        
        self._executor.shutdown(wait=wait)
        logger.info("WorkerPool shutdown complete")


# =============================================================================
# TASK HANDLERS - Register these with the WorkerPool
# =============================================================================

def handler_extract_file(data: Dict, progress_callback: Callable) -> Dict:
    """
    Handler for single file extraction.
    
    data: {"file_path": str, "scanner": CognitiveScanner}
    """
    from pathlib import Path
    
    file_path = Path(data["file_path"])
    scanner = data.get("scanner")
    
    if not scanner:
        raise ValueError("Scanner not provided")
    
    progress_callback(0, 1, f"Extraindo {file_path.name}...")
    
    # Call scanner's extract method
    result = scanner.process_single_file(file_path)
    
    progress_callback(1, 1, f"Concluído: {file_path.name}")
    
    return {
        "file_path": str(file_path),
        "extraction_result": result
    }


def handler_extract_batch(data: Dict, progress_callback: Callable) -> Dict:
    """
    Handler for batch file extraction.
    
    data: {"files": List[str], "scanner": CognitiveScanner}
    """
    from pathlib import Path
    
    files = [Path(f) for f in data["files"]]
    scanner = data.get("scanner")
    
    if not scanner:
        raise ValueError("Scanner not provided")
    
    results = []
    total = len(files)
    
    for i, file_path in enumerate(files):
        progress_callback(i, total, f"Processando {file_path.name}...")
        
        try:
            result = scanner.process_single_file(file_path)
            results.append({"file": str(file_path), "success": True, "data": result})
        except Exception as e:
            results.append({"file": str(file_path), "success": False, "error": str(e)})
    
    progress_callback(total, total, f"Concluído: {total} arquivos")
    
    return {
        "processed": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }


def handler_import_files(data: Dict, progress_callback: Callable) -> Dict:
    """
    Handler for file import operation.
    
    data: {"files": List[str], "scanner": CognitiveScanner, "db": SRDADatabase}
    """
    from pathlib import Path
    
    files = [Path(f) for f in data["files"]]
    scanner = data.get("scanner")
    db = data.get("db")
    
    if not scanner or not db:
        raise ValueError("Scanner and DB required")
    
    results = []
    total = len(files)
    
    for i, file_path in enumerate(files):
        progress_callback(i, total, f"Importando {file_path.name}...")
        
        try:
            # Process file
            result = scanner.process_single_file(file_path)
            
            # Insert to DB
            if result and result.get("document"):
                doc_id = db.insert_document(result["document"])
                results.append({
                    "file": str(file_path),
                    "success": True,
                    "doc_id": doc_id
                })
            else:
                results.append({
                    "file": str(file_path),
                    "success": False,
                    "error": "No document data extracted"
                })
                
        except Exception as e:
            results.append({
                "file": str(file_path),
                "success": False,
                "error": str(e)
            })
    
    progress_callback(total, total, f"Importação concluída: {total} arquivos")
    
    return {
        "imported": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }


def handler_gemini_batch(data: Dict, progress_callback: Callable) -> Dict:
    """
    Handler for Gemini batch processing.
    
    data: {"files": List[str], "oracle": GeminiOracle}
    """
    from pathlib import Path
    
    files = [Path(f) for f in data["files"]]
    oracle = data.get("oracle")
    
    if not oracle:
        raise ValueError("GeminiOracle not provided")
    
    # TODO: Implement batch processing in GeminiOracle
    # For now, process sequentially
    results = []
    total = len(files)
    
    for i, file_path in enumerate(files):
        progress_callback(i, total, f"Gemini: {file_path.name}...")
        
        try:
            result = oracle.process_document(file_path)
            results.append({
                "file": str(file_path),
                "success": result.success,
                "data": result.final_data if result.success else None,
                "error": result.error if not result.success else None
            })
        except Exception as e:
            results.append({
                "file": str(file_path),
                "success": False,
                "error": str(e)
            })
    
    progress_callback(total, total, f"Gemini concluído: {total} arquivos")
    
    return {
        "processed": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "results": results
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_worker_pool(tk_root=None, max_workers: int = 4) -> WorkerPool:
    """
    Create and configure a WorkerPool with all standard handlers.
    
    Args:
        tk_root: Tkinter root for safe UI updates
        max_workers: Max concurrent workers
        
    Returns:
        Configured WorkerPool instance
    """
    pool = WorkerPool(max_workers=max_workers, tk_root=tk_root)
    
    # Register all handlers
    pool.register_handler(TaskType.EXTRACT_FILE, handler_extract_file)
    pool.register_handler(TaskType.EXTRACT_BATCH, handler_extract_batch)
    pool.register_handler(TaskType.IMPORT_FILES, handler_import_files)
    pool.register_handler(TaskType.GEMINI_BATCH, handler_gemini_batch)
    
    return pool
