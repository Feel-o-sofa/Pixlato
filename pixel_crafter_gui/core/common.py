"""
Pixlato - Common Utilities & Task Management
============================================
Handles asynchronous task lifecycle, interruption, and global ID registry.
"""

import threading
import sys

# Detection for packaged environment (e.g., PyInstaller)
IS_PACKAGED = getattr(sys, 'frozen', False)

def debug_log(message):
    """Prints log messages only in development environment."""
    if not IS_PACKAGED:
        print(message)

class OperationCancelled(Exception):
    """Exception raised when a long-running operation is cancelled by the user."""
    pass

class TaskManager:
    """
    Singleton-style manager to track the active task ID.
    Enables cooperative interruption across different modules.
    """
    _global_active_id = 0
    _lock = threading.Lock()

    @classmethod
    def start_new_task(cls):
        """
        Increments and returns a new global task ID.
        This invalidates all previous tasks.
        """
        with cls._lock:
            cls._global_active_id += 1
            return cls._global_active_id

    @classmethod
    def interrupt_all(cls):
        """
        Forcefully increments the global ID to cancel any running interruptible tasks.
        """
        with cls._lock:
            cls._global_active_id += 1

    @classmethod
    def check(cls, assigned_id):
        """
        Checks if the task with assigned_id is still the active one.
        If not, raises OperationCancelled to stop execution.
        """
        if assigned_id is None:
            return  # Tasks without ID are not interruptible
            
        with cls._lock:
            is_valid = (assigned_id == cls._global_active_id)
            
        if not is_valid:
            raise OperationCancelled(f"Task {assigned_id} is no longer active (Current: {cls._global_active_id})")

    @classmethod
    def get_active_id(cls):
        with cls._lock:
            return cls._global_active_id
