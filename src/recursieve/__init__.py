from importlib.metadata import PackageNotFoundError, version

from .recursieve import recursieve

try:
    __version__ = version("recursieve")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["recursieve", "__version__"]
