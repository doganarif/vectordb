from contextlib import contextmanager
from threading import Condition, Lock
from typing import Iterator


class ReaderWriterLock:
    def __init__(self) -> None:
        self._mutex = Lock()
        self._cond = Condition(self._mutex)
        self._active_readers = 0
        self._writer_active = False
        self._waiting_writers = 0

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer_active or self._waiting_writers > 0:
                self._cond.wait()
            self._active_readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._active_readers -= 1
            if self._active_readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._waiting_writers += 1
            while self._writer_active or self._active_readers > 0:
                self._cond.wait()
            self._waiting_writers -= 1
            self._writer_active = True

    def release_write(self) -> None:
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()

    @contextmanager
    def read_lock(self) -> Iterator[None]:
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()
