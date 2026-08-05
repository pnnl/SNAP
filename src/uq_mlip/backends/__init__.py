"""Backend adapter registry for flagship MLIPs."""

from uq_mlip.backends.chgnet import CHGNetExtractor
from uq_mlip.backends.mace import MACEExtractor
from uq_mlip.backends.uma import UMAExtractor


def get_extractor(backend: str, **kwargs):
    """Create an embedding extractor for a supported backend."""

    normalized = backend.lower()
    if normalized == "mace":
        return MACEExtractor(**kwargs)
    if normalized == "uma":
        return UMAExtractor(**kwargs)
    if normalized == "chgnet":
        return CHGNetExtractor(**kwargs)
    raise ValueError(
        f"Unsupported backend '{backend}'. Supported backends: mace, uma, chgnet."
    )


__all__ = ["CHGNetExtractor", "MACEExtractor", "UMAExtractor", "get_extractor"]
