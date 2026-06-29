"""NOK/USD forecasting from the oil-futures term structure.

The package is split into modules that can be run step by step:
    config        - paths and shared parameters
    data_loader   - modular interface to data sources (swap the source without
                    touching the rest of the code)
"""
import sys as _sys

# Some Windows consoles default to cp1252 and crash on non-ASCII characters in
# print(). Force UTF-8 on stdout/stderr so modules can print freely.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
