"""Analysis of raman spectra from MIT scope."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("raman-analysis")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Ian Hunt-Isaak"
__email__ = "ianhuntisaak@gmail.com"
