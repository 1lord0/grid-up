"""Mevcut submission'da yalnız doğrulanmış cold-start satırlarını değiştirir."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    cold = pd.read_csv(args.cold)
    columns = ["id", "tuketim"]

    if list(base.columns) != columns or list(cold.columns) != columns:
        raise ValueError("Beklenmeyen CSV kolonları")
    if base["id"].duplicated().any() or cold["id"].duplicated().any():
        raise ValueError("Yinelenen ID bulundu")
    if cold["tuketim"].isna().any() or not np.isfinite(cold["tuketim"]).all():
        raise ValueError("Cold-start tahmininde geçersiz değer bulundu")
    if (cold["tuketim"] < 0).any():
        raise ValueError("Cold-start tahmininde negatif değer bulundu")

    base_ids = pd.Index(base["id"])
    cold_ids = pd.Index(cold["id"])
    if len(cold_ids.difference(base_ids)):
        raise ValueError("Cold-start dosyasında base submission'da olmayan ID var")

    original = base["tuketim"].to_numpy(copy=True)
    cold_mask = base["id"].isin(cold_ids).to_numpy()
    replacements = cold.set_index("id")["tuketim"]
    base.loc[cold_mask, "tuketim"] = base.loc[cold_mask, "id"].map(replacements)

    if int(cold_mask.sum()) != len(cold):
        raise ValueError("Cold-start satırları tam olarak birer kez değiştirilemedi")
    if not np.array_equal(base.loc[~cold_mask, "tuketim"].to_numpy(), original[~cold_mask]):
        raise AssertionError("Cold-start dışındaki bir değer değişti")
    if base["tuketim"].isna().any() or not np.isfinite(base["tuketim"]).all():
        raise ValueError("Çıktıda geçersiz değer bulundu")
    if (base["tuketim"] < 0).any():
        raise ValueError("Çıktıda negatif değer bulundu")

    base.to_csv(args.output, index=False)
    print(f"satır={len(base):,}")
    print(f"değişen_cold={int(cold_mask.sum()):,}")
    print(f"değişmeyen={int((~cold_mask).sum()):,}")
    print(f"sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
