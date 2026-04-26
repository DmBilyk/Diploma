import os
import importlib.util
import inspect
import logging
from typing import Dict, Type
from app.plugins.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Manager class responsible for discovering and loading optimization plugins.
    """

    def __init__(self, plugins_dir: str = None):
        """
        Initializes the PluginManager.

        Args:
            plugins_dir (str, optional): The directory to scan for plugins.
                                         Defaults to the directory of this file.
        """
        self.plugins_dir = plugins_dir or os.path.dirname(os.path.abspath(__file__))
        self._cache: Dict[str, Type[BaseOptimizer]] | None = None

    def get_plugins(self) -> Dict[str, Type[BaseOptimizer]]:
        """
        Scans the plugins folder for .py files, imports them, and extracts classes
        that inherit from BaseOptimizer.

        Returns:
            Dict[str, Type[BaseOptimizer]]: A dictionary mapped as {"Algorithm name": Class}.
        """
        if self._cache is not None:
            return self._cache

        discovered_plugins: Dict[str, Type[BaseOptimizer]] = {}

        if not os.path.exists(self.plugins_dir):
            logger.warning(f"Plugins folder not found at: {self.plugins_dir}")
            return discovered_plugins

        for filename in os.listdir(self.plugins_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                # Skip the core plugin system files
                if filename in ("base_optimizer.py", "plugin_manager.py"):
                    continue

                module_name = filename[:-3]
                file_path = os.path.join(self.plugins_dir, filename)

                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        # Execute module in its own namespace
                        spec.loader.exec_module(module)

                        # Inspect module for BaseOptimizer subclasses
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            # issubclass() перевіряє весь MRO (а не лише прямих батьків),
                            # порівнює реальні об'єкти класів, а не рядки імен.
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