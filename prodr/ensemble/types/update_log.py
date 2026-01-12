from typing import TypedDict

from .events import InsertionEvent, SplitEvent


class UpdateLog(TypedDict):
    """
    Log of updates (insertions and splits) made to the tree during an insertion operation.
    """

    insertion_log: list[InsertionEvent]
    split_log: list[SplitEvent]
