import logging
from typing import Any, Dict


class ExtensionBase:
    """Base Class for all extensions

    Other extensions should inherit from this class and implement the run method

    Parameters
    ----------
    context : dict
        context dictionary that holds the data and other information
    settings : dict
        settings dictionary
    logger : logging.Logger
        logger for the extension
    """

    def __init__(self, context: Dict[str, Any], settings: Dict[str, Any], logger: logging.Logger) -> None:
        self.context = context
        self.settings = settings
        self.logger = logger

    def run(self) -> None:
        raise NotImplementedError("run method not implemented")
