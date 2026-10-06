import platform
from importlib.metadata import PackageNotFoundError, version


def _package_available(package_name: str) -> bool:
    """Check if a package is installed.

    importlib.metadata (stdlib), not pkg_resources: pkg_resources lives in setuptools, which Python 3.12 venvs do not ship.

    :param package_name: The name of the package to be checked.
    :return: `True` if the package is available. `False` otherwise.
    """
    try:
        version(package_name)
        return True
    except PackageNotFoundError:
        return False


_IS_WINDOWS = platform.system() == "Windows"

_SH_AVAILABLE = not _IS_WINDOWS and _package_available("sh")
_WANDB_AVAILABLE = _package_available("wandb")
