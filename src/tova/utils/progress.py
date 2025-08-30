import logging
from dataclasses import dataclass
from typing import Callable, Optional

ProgressCallback = Callable[[float, str], None]  # shortcut: (progress, message)

@dataclass
class ProgressReporter:
    callback: Optional[ProgressCallback] = None
    start: float = 0.0
    end: float = 1.0
    logger: logging.Logger = logging.getLogger(__name__)

    def report(self, progress: float, message: str = "") -> None:
        """Report progress to the callback, if it exists, and shows a message in the logger (message from the caller + progress in %)
        """
        self.logger.info(f"Progress: {progress*100:.2f}%, Message: {message}")
        if not self.callback:
            return
        
        progress = max(0.0, min(1.0, progress))  # clamp to [0, 1]
        progress_global = self.start + (self.end - self.start) * progress
        try:
            self.callback(progress_global, message)
        except Exception as e:
            self.logger.error(f"Error reporting progress: {e}")
            pass
        
    def report_subrange(self, sub_start: float, sub_end: float) -> 'ProgressReporter':
        """Create a new ProgressReporter for a sub-range of the current progress range.
        """
        sub_start = max(0.0, min(1.0, sub_start))
        sub_end = max(0.0, min(1.0, sub_end))
        
        new_start = self.start + (self.end - self.start) * sub_start
        new_end = self.start + (self.end - self.start) * sub_end
        
        return ProgressReporter(callback=self.callback, start=new_start, end=new_end, logger=self.logger)