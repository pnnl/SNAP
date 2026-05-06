"""Backend adapter registry for flagship MLIPs."""

from uq_mlip.backends.mace import MACEExtractor
from uq_mlip.backends.uma import UMAExtractor


def get_extractor(backend: str, **kwargs):
    """Create an embedding extractor for a supported backend."""

    normalized = backend.lower()
    if normalized == "mace":
        return MACEExtractor(**kwargs)
    if normalized == "uma":
        return UMAExtractor(**kwargs)
    raise ValueError(f"Unsupported backend '{backend}'. Supported backends: mace, uma.")


__all__ = ["MACEExtractor", "UMAExtractor", "get_extractor"]
