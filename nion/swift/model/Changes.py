from __future__ import annotations

import abc
import pathlib
import typing

if typing.TYPE_CHECKING:
    from nion.swift.model import DocumentModel


class UndeleteRefusedError(Exception):
    """An undelete refused because the file at file_path can't be written or moved, which a later undelete may try again."""

    def __init__(self, file_path: pathlib.Path) -> None:
        super().__init__(f"Unable to write or move data file {file_path}")
        self.file_path = file_path


class UndeleteBase(abc.ABC):

    @abc.abstractmethod
    def close(self) -> None: ...

    def check_undelete(self, document_model: DocumentModel.DocumentModel) -> None:
        """Raise UndeleteRefusedError if undelete would be refused."""

    @abc.abstractmethod
    def undelete(self, document_model: DocumentModel.DocumentModel) -> None: ...


class UndeleteLog:
    count = 0  # useful for detecting leaks in tests

    def __init__(self) -> None:
        UndeleteLog.count += 1
        self.__items : typing.List[UndeleteBase] = list()
        self.__is_closed = False

    def close(self) -> None:
        # a command whose undo is refused keeps its logs, and closes the ones already undeleted again.
        if not self.__is_closed:
            self.__is_closed = True
            for item in self.__items:
                item.close()
            self.__items = typing.cast(typing.Any, None)
            UndeleteLog.count -= 1

    def append(self, item: UndeleteBase) -> None:
        self.__items.append(item)

    def undelete_all(self, document_model: DocumentModel.DocumentModel) -> None:
        """Undelete the items, or raise UndeleteRefusedError with none of them undeleted.

        A closed log has undeleted its items already, in an undo which a later log refused and which is now tried again.
        """
        if not self.__is_closed:
            # every entry is checked first, since a display item may undelete before its data item, and a refusal
            # between them would leave the display item without its data item.
            for entry in self.__items:
                entry.check_undelete(document_model)
            for entry in reversed(self.__items):
                entry.undelete(document_model)

    @property
    def _items(self) -> typing.List[UndeleteBase]:
        return self.__items
