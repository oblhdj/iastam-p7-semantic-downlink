import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_collection_modifyitems(items):
    """Tiers (code/pytest.ini): a test that runs the detector is marked `inference`, one that
    renders the dashboard `page`; everything else is `unit` and runs anywhere."""
    import pytest
    for item in items:
        if not (item.get_closest_marker("inference") or item.get_closest_marker("page")):
            item.add_marker(pytest.mark.unit)
