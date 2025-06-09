"""
Неблокирующий Parquet-логер: SimPy-процесс читает события из Queue
и пакетами пишет на диск.  Для тестов <10^6 строк хватит и памяти,
но лучше писать ин-крементно.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ── единый формат записи ──────────────────────────────────────────────────────
@dataclass
class EventRow:
    sim_time: float       # секунды от T0
    pid: int              # id разработчика
    sp_done: float        # story-points выполнено за тик
    tp_end: float         # усталость после тика
    sprint: int           # номер спринта (заполняет SprintPlanner)


class ParquetLogger:
    """
    Фоновый процесс SimPy::  done_q → batch → parquet.
    Путь <root>/events-000.parquet, 001, 002, …
    """

    def __init__(self, root: Path, batch_size: int = 5000) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self._buffer: List[Dict[str, Any]] = []
        self._file_no = 0
        self._schema = pa.schema([
            ("sim_time", pa.float64()),
            ("pid", pa.int16()),
            ("sp_done", pa.float32()),
            ("tp_end", pa.float32()),
            ("sprint", pa.int32()),
        ])

    # SimPy process — вызывается из run_simulation.py
    def collect(self, done_q):
        """Coroutine: получает dict-и, складывает, пишет партиями."""
        while True:
            row: Dict[str, Any] = yield done_q.get()
            self._buffer.append(row)

            if len(self._buffer) >= self.batch_size:
                self._flush()

    # ── API для пост-анализа ───────────────────────────────────────────────────
    def to_dataframe(self) -> pd.DataFrame:
        """Читает все parquet-файлы + хвостовой buffer → pandas DF."""
        self._flush()  # финальный дроп

        frames = [
            pq.read_table(path).to_pandas()
            for path in sorted(self.root.glob("events-*.parquet"))
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ── внутреннее ────────────────────────────────────────────────────────────
    def _flush(self) -> None:
        if not self._buffer:
            return
        batch = pa.Table.from_pylist(self._buffer, schema=self._schema)
        fname = self.root / f"events-{self._file_no:03d}.parquet"
        pq.write_table(batch, fname, compression="zstd")
        self._file_no += 1
        self._buffer.clear()
