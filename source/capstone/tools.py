#!/usr/bin/env python3
import os.path
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from traceback import extract_tb, format_exception_only
from typing import Generator

import orjson
import pandas as pd
import zmq
import zmq.asyncio

warn = warnings.warn

def tprint(*args, **kwargs):
    """ Wrapper for printing with threads """
    print(*args, flush=True, **kwargs)


class ZmqListener:
    """ Listener class for ZMQ Port """

    def __init__(self, verbose: int, port: int = 5555) -> None:
        self.port = port
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')

        self.verbose = verbose
        self.buffer = pd.DataFrame()

    async def listen(self) -> Generator[pd.DataFrame, None, None]:
        while True:
            message = await self.socket.recv()
            msg_dict = orjson.loads(message)
            df_line = pd.DataFrame(msg_dict['xyzv'], columns=['x', 'y', 'z', 'v']).assign(ts=msg_dict['ts'], frame=msg_dict['frame'])
            # currently disabling pyarrow backend bc of weird error
            # <EOL>.convert_dtypes(dtype_backend='pyarrow', inplace=True)
            yield df_line

    def append_to_buffer(self, df_line: pd.DataFrame) -> None:
        self.buffer = pd.concat([self.buffer, df_line], ignore_index=True)


''' OLD ZMQ CLASS
class ZmqListener:
    """ Listener class for ZMQ Port """

    def __init__(self, verbose: int, port: int = 5555):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')

        self.verbose = verbose
        self.buffer = pd.DataFrame()

    def listen(self) -> pd.DataFrame:
        while True:
            message = self.socket.recv()
            msg_dict = orjson.loads(message)
            df_line = pd.DataFrame(msg_dict['xyzv'], columns=['x', 'y', 'z', 'v']).assign(ts=msg_dict['ts'], frame=msg_dict['frame'])
            # currently disabling pyarrow backend bc of weird error
            # <EOL>.convert_dtypes(dtype_backend='pyarrow', inplace=True)
            yield df_line

    def append_to_buffer(self, df_line: pd.DataFrame):
        self.buffer = pd.concat([self.buffer, df_line], ignore_index=True)
'''


class DebugThreadPoolExecutor(ThreadPoolExecutor):
    """ Wrapper for ThreadPoolExecutor that delivers Warnings & Exceptions from Threads """

    def __init__(self, *args, non_blocking: bool, verbose: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.non_blocking = non_blocking
        self.verbose = verbose
        self._exceptions_queue = deque()
        self._warnings_queue = deque()

    def submit(self, fn, *args, **kwargs):
        def wrapped_fn(*args, **kwargs):
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter('always')
                try:
                    result = fn(*args, **kwargs)
                except Exception as e:
                    self._exceptions_queue.append(e)
                else:
                    for w in captured_warnings:
                        if 'hdbscan' not in w.category.__module__:
                            self._warnings_queue.append(w)
                    return result

        future = super().submit(wrapped_fn, *args, **kwargs)
        self.raise_exception()
        return future

    def raise_exception(self):
        while self._exceptions_queue:
            e = self._exceptions_queue.popleft()
            tb = extract_tb(e.__traceback__)[-1]
            exc_info = ''.join(format_exception_only(type(e), e)).strip()
            if self.non_blocking:
                # Print the exception without ending the program
                tprint(f"Exception occurred: '{exc_info}' => See: {os.path.basename(tb.filename)}:{tb.lineno}")
            else:
                # Raise the exception and end the program
                raise e
        if self.verbose > 0:
            while self._warnings_queue:
                w = self._warnings_queue.popleft()
                tprint(f"{pd.Timestamp.now().strftime('%X')} | WARN: {w.category.__name__}, '{w.message}'")
