import csv
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Any, Iterable

def save_csv_results(
    predicted_targets: Dict[int, List[Mapping[str, Any]]],
    path: str = "csv_results",
    *,
    clean: bool = True,
    sort: bool = True,
) -> None:
    """
    Write per-sample picks to CSV files.

    Input format:
        predicted_targets = {
            <sample_id>: [
                {"channel": <int>, "time_index": <int>, "type": <str 'P'|'S'|...>},
                ...
            ],
            ...
        }

    Output:
        Creates `<path>/<sample_id>.csv` for each sample, with header:
            channel_index,phase_index,phase_type

    Args:
        predicted_targets: Mapping from sample id to a list of pick dicts.
        path: Output directory for CSV files.
        clean: If True, deletes the output directory before writing.
        sort: If True, sorts rows by (channel, time_index) for deterministic output.
    """
    out_dir = Path(path)

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = ["channel_index", "phase_index", "phase_type"]

    for sample_id, rows in predicted_targets.items():
        rows = rows or []

        if sort:
            def _key(r: Mapping[str, Any]):
                ch = int(r.get("channel", 0))
                ti = int(r.get("time_index", 0))
                tp = str(r.get("type", ""))
                return (ch, ti, tp)
            rows = sorted(rows, key=_key)

        csv_path = out_dir / f"{int(sample_id)}.csv"
        tmp_path = csv_path.with_suffix(".csv.tmp")

        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for r in rows:
                ch = int(r.get("channel", 0))
                ti = int(r.get("time_index", 0))
                tp = str(r.get("type", ""))
                writer.writerow([ch, ti, tp])

        tmp_path.replace(csv_path)
