import os
import importlib.util
import inspect
import logging
from typing import Dict, Type
from app.plugins.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class PluginManager:
    """Discovers and loads ``BaseOptimizer`` subclasses from the plugins folder."""

    def __init__(self, plugins_dir: str = None):
        """Initialise the manager.

        Parameters
        ----------
        plugins_dir : str, optional
            Directory to scan. Defaults to the directory of this file.
        """
        self.plugins_dir = plugins_dir or os.path.dirname(os.path.abspath(__file__))
        self._cache: Dict[str, Type[BaseOptimizer]] | None = None

    def get_plugins(self) -> Dict[str, Type[BaseOptimizer]]:
        """Scan the plugins folder and return discovered optimisers.

        Returns
        -------
        Dict[str, Type[BaseOptimizer]]
            Mapping ``{class_name: class}`` for every ``BaseOptimizer`` subclass.
        """
        if self._cache is not None:
            return self._cache

        discovered_plugins: Dict[str, Type[BaseOptimizer]] = {}

        if not os.path.exists(self.plugins_dir):
            logger.warning(f"Plugins folder not found at: {self.plugins_dir}")
            return discovered_plugins

        for filename in os.listdir(self.plugins_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                if filename in ("base_optimizer.py", "plugin_manager.py"):
                    continue

                module_name = filename[:-3]
                file_path = os.path.join(self.plugins_dir, filename)

                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            # issubclass walks the full MRO and compares class objects, not names.
                            if (
                                    issubclass(obj, BaseOptimizer)
                                    and obj is not BaseOptimizer
                                    and obj.__module__ == module_name
                            ):
                                discovered_plugins[name] = obj

                except Exception as e:
                    logger.error(f"Error loading plugin from {filename}: {e}")

        self._cache = discovered_plugins
        return discovered_plugins
