from typing import Callable, Optional, Tuple, Union


class StockStatsError(Exception):
    pass


_dft_windows = {
    # sort alphabetically
    'ao': (5, 34),
    'aroon': 25,
    'atr': 14,
    'boll': 20,
    'cci': 14,
    'change': 1,
    'chop': 14,
    'cmo': 14,
    'coppock': (10, 11, 14),
    'cr': 26,
    'cti': 12,
    'dma': (10, 50),
    'eri': 13,
    'eribear': 13,
    'eribull': 13,
    'ichimoku': (9, 26, 52),
    'inertia': (20, 14),
    'ftr': 9,
    'kama': (10, 5, 34),  # window, fast, slow
    'kdjd': 9,
    'kdjj': 9,
    'kdjk': 9,
    'ker': 10,
    'macd': (12, 26, 9),  # short, long, signal
    'mfi': 14,
    'ndi': 14,
    'pdi': 14,
    'pgo': 14,
    'ppo': (12, 26, 9),  # short, long, signal
    'pvo': (12, 26, 9),  # short, long, signal
    'psl': 12,
    'qqe': (14, 5),  # rsi, rsi ema
    'rsi': 14,
    'rsv': 9,
    'rvgi': 14,
    'stochrsi': 14,
    'supertrend': 14,
    'tema': 5,
    'trix': 12,
    'wr': 14,
    'wt': (10, 21),
    'vr': 26,
    'vwma': 14,
    'num': 0,
}


_window_listeners: list[Callable[[str, Tuple[int, ...] | None], None]] = []


def register_window_listener(
        listener: Callable[[str, Tuple[int, ...] | None], None]
):
    """Register a callback invoked whenever a default window changes."""
    _window_listeners.append(listener)


def _notify_window_listeners(name: str):
    value = get_default_windows_tuple(name)
    for listener in _window_listeners:
        listener(name, value)


def set_dft_window(name: str, windows: Union[int, tuple[int, ...]]):
    ret = _dft_windows.get(name)
    _dft_windows[name] = windows
    _notify_window_listeners(name)
    return ret


_dft_column = {
    # sort alphabetically
    'cti': 'close',
    'dma': 'close',
    'kama': 'close',
    'ker': 'close',
    'psl': 'close',
    'tema': 'close',
    'trix': 'close',
}


def get_default_windows_tuple(name: str) -> Optional[Tuple[int, ...]]:
    dft = _dft_windows.get(name)
    if dft is None:
        return None
    if isinstance(dft, int):
        return (dft,)
    return tuple(dft)


def dft_windows(name: str) -> Optional[str]:
    dft = get_default_windows_tuple(name)
    if dft is None:
        return None
    return ','.join(map(str, dft))


def dft_column(name: str) -> Optional[str]:
    return _dft_column.get(name)


def get_default_column(name: str) -> Optional[str]:
    return dft_column(name)
