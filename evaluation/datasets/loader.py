import pandas as pd
from pathlib import Path
from typing import List, Union

RENAME_COLUMN = {"No": "no", "Query": "query", "Expected_output": "expected_output"}

def load_dataset(path: Union[str, Path]) -> List[dict]:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Query", "Expected_output"]).rename(columns=RENAME_COLUMN)
    if "query_type" not in df.columns:
        df["query_type"] = ""
    rows = df[["no", "query", "query_type", "expected_output"]].to_dict("records")
    for r in rows:
        r["no"] = str(r["no"]).removesuffix(".0")
        r["query"] = str(r["query"]).strip()
        r["query_type"] = str(r["query_type"]).strip()
        r["expected_output"] = str(r["expected_output"]).strip()
    return rows