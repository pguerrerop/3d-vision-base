from abc import ABC, abstractmethod
from pathlib import Path


class AcquisitionBase(ABC):
    """Base class for acquisition modes (offline PLY, Harvesters, FTP, …)."""

    @abstractmethod
    def acquire(self, *args, **kwargs) -> tuple[str, Path]:
        """Run acquisition and return (take_id, published_folder)."""
        raise NotImplementedError
