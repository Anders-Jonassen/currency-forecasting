"""NOK/USD-prognose fra oljefuturesenes terminstruktur.

Pakken er delt i moduler som kan kjøres steg for steg:
    config        – stier og felles parametre
    data_loader   – modulært grensesnitt mot datakilder (bytt kilde uten å
                    endre resten av koden)
"""
import sys as _sys

# Windows-konsollen er ofte cp1252 og krasjer på tegn som 'λ'/'å' i print().
# Vi tvinger UTF-8 på stdout/stderr slik at modulene kan skrive norske tegn fritt.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
