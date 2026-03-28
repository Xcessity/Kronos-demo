import re
from pathlib import Path

import pandas as pd

_LEGACY = re.compile(
    r"^(close_mean|close_std|volume_mean|volume_std|actual_close|upside_probability)_h(\d+)$"
)


class EvaluationResults:
    """Loader for evaluation_results.csv that handles legacy headers without a timeframe token.

    Legacy format:  close_mean_h1, close_std_h1, ...
    Current format: close_mean_2h_h1, close_std_2h_h1, ...

    When a legacy file is detected the user is prompted for the timeframe, the
    columns are renamed in-place, and the file is overwritten so the migration
    only happens once.
    """

    @staticmethod
    def load(csv_path: Path, parse_dates: bool = True) -> pd.DataFrame:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path, parse_dates=["timestamp"] if parse_dates else False)

        legacy_cols = [c for c in df.columns if _LEGACY.match(c)]
        if not legacy_cols:
            return df

        print(f"\n[EvaluationResults] Legacy column headers detected in {csv_path.name}")
        print(f"  e.g. {', '.join(legacy_cols[:3])}")
        timeframe = input("  Enter candle timeframe to encode (e.g. 1h, 2h, 15m): ").strip()
        if not timeframe:
            raise ValueError("Timeframe is required to migrate legacy headers.")

        rename = {}
        for col in df.columns:
            m = _LEGACY.match(col)
            if m:
                metric, horizon = m.group(1), m.group(2)
                rename[col] = f"{metric}_{timeframe}_h{horizon}"

        df.rename(columns=rename, inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"  Renamed {len(rename)} columns and saved updated headers to {csv_path.name}\n")
        return df
