"""CSV logging utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class CSVLogger:
    output_dir: Path
    name: str

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, float]] = []

    def log(self, row: Dict[str, float]) -> None:
        self.rows.append(dict(row))

    def flush(self) -> Path:
        path = self.output_dir / f"{self.name}.csv"
        df = pd.DataFrame(self.rows)
        df.to_csv(path, index=False)
        return path
