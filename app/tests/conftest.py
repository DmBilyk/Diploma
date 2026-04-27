from pathlib import Path
import pytest

def pytest_collection_modifyitems(config, items):
    for item in items:
        path = Path(item.fspath)

        if "unit" in path.parts:
            item.add_marker(pytest.mark.unit)

        elif "component" in path.parts:
            item.add_marker(pytest.mark.component)

        elif "integration" in path.parts:
            item.add_marker(pytest.mark.integration)

        elif "e2e" in path.parts:
            item.add_marker(pytest.mark.e2e)