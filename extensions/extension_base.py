import logging
from typing import Any, Dict


class ExtensionBase:
    def __init__(self, context: Dict[str, Any], settings: Dict[str, Any], logger: logging.Logger) -> None:
        self.context = context
        self.settings = settings
        self.logger = logging

    def run(self) -> None:
        raise NotImplementedError("run method not implemented")
