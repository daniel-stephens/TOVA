import logging
from dataclasses import dataclass, field
from threading import Event
from typing import Optional

@dataclass
class CancellationToken:
    _ev: Event = field(default_factory=Event)

    def cancel(self) -> None:
        self._ev.set()

    def is_cancelled(self) -> bool:
        return self._ev.is_set()

class CancelledError(RuntimeError):
    pass


def check_cancel(
    cancel: "CancellationToken | None",
    logger: Optional[logging.Logger]
) -> None:
    """Function to avoid repeating the following call:
        if cancel and cancel.is_cancelled(): raise CancelledError()"
    """
    if cancel and cancel.is_cancelled():
        if logger:
            logger.info("Operation was cancelled")
        raise CancelledError("Operation was cancelled")