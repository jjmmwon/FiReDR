from typing import TypedDict

from .events import InsertionEvent, SplitEvent


class UpdateLog(TypedDict):
    insertion_log: list[InsertionEvent]
    split_log: list[SplitEvent]
