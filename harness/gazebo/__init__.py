"""Independent Gazebo pick-cell demonstration.

Run ``python -m harness.gazebo.cli --help`` for the operator commands or
``streamlit run harness/gazebo/console.py --server.port 8502`` for the local
console.  This package deliberately has no product imports.
"""

__all__ = ["PickCellController"]


def __getattr__(name: str):
    if name == "PickCellController":
        from .controller import PickCellController

        return PickCellController
    raise AttributeError(name)
