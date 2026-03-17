"""Download both public datasets into data/raw/.

ASOS Digital Experiments Dataset (Liu et al., NeurIPS 2021), hosted on OSF: https://osf.io/64jsb/
Criteo Uplift Prediction Dataset v2.1 (Diemert et al., 2018), hosted on Hugging Face.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OSF_LISTING = "https://api.osf.io/v2/nodes/64jsb/files/osfstorage/"
ASOS_FILE = "asos_digital_experiments_dataset.parquet"
CRITEO_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"exists: {dest.name}")
        return
    print(f"downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.rename(dest)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(OSF_LISTING) as response:
        files = json.load(response)["data"]
    asos_url = next(f["links"]["download"] for f in files if f["attributes"]["name"] == ASOS_FILE)
    _download(asos_url, RAW / ASOS_FILE)
    _download(CRITEO_URL, RAW / "criteo-research-uplift-v2.1.csv.gz")


if __name__ == "__main__":
    main()
