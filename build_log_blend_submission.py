"""İki submission'ın yalnız belirtilen satırlarını RMSLE-uyumlu log uzayında karıştırır."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.weight <= 1:
        raise ValueError("Ağırlık 0 ile 1 arasında olmalıdır")

    base = pd.read_csv(args.base)
    challenger = pd.read_csv(args.challenger)
    cold = pd.read_csv(args.cold, usecols=["id"])
    expected = ["id", "tuketim"]

    if list(base.columns) != expected or list(challenger.columns) != expected:
        raise ValueError("Beklenmeyen submission kolonları")
    if len(base) != len(challenger) or not base["id"].equals(challenger["id"]):
        raise ValueError("Submission ID sıraları eşleşmiyor")
    if cold["id"].duplicated().any():
        raise ValueError("Cold-start ID listesinde tekrar var")

    cold_mask = base["id"].isin(cold["id"]).to_numpy()
    if int(cold_mask.sum()) != len(cold):
        raise ValueError("Cold-start ID kümesi submission ile eşleşmiyor")

    output = base.copy()
    base_cold = base.loc[cold_mask, "tuketim"].to_numpy()
    challenger_cold = challenger.loc[cold_mask, "tuketim"].to_numpy()
    blended = np.expm1(
        (1 - args.weight) * np.log1p(base_cold)
        + args.weight * np.log1p(challenger_cold)
    )
    output.loc[cold_mask, "tuketim"] = np.maximum(blended, 0)

    if not np.array_equal(
        output.loc[~cold_mask, "tuketim"].to_numpy(),
        base.loc[~cold_mask, "tuketim"].to_numpy(),
    ):
        raise AssertionError("Cold-start dışındaki tahminler değişti")
    if output["tuketim"].isna().any() or not np.isfinite(output["tuketim"]).all():
        raise ValueError("Çıktıda geçersiz tahmin var")

    # Base dosyanın cold-start dışındaki satırlarını metin seviyesinde de aynen
    # koru. Böylece pandas'ın kayan nokta yeniden yazımı bile bu satırlara
    # dokunamaz.
    replacement_map = dict(zip(base.loc[cold_mask, "id"], blended, strict=True))
    with args.base.open("r", encoding="utf-8", newline="") as source, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as sink:
        sink.write(source.readline())
        for line in source:
            body = line.rstrip("\r\n")
            newline = line[len(body) :]
            row_id, _ = body.split(",", 1)
            if row_id in replacement_map:
                sink.write(f"{row_id},{replacement_map[row_id]:.17g}{newline}")
            else:
                sink.write(line)

    written = pd.read_csv(args.output)
    if not written["id"].equals(base["id"]):
        raise AssertionError("Yazılan dosyanın ID sırası değişti")
    if not np.array_equal(
        written.loc[~cold_mask, "tuketim"].to_numpy(),
        base.loc[~cold_mask, "tuketim"].to_numpy(),
    ):
        raise AssertionError("Yazılan dosyada cold-start dışı değer değişti")
    print(f"ağırlık={args.weight:.6f}")
    print(f"karıştırılan_cold_satırı={int(cold_mask.sum()):,}")
    print(f"değişmeyen_satır={int((~cold_mask).sum()):,}")
    print(f"cold_medyan={np.median(blended):.6f}")
    print(f"cold_ortalama={np.mean(blended):.6f}")
    print(f"sha256={file_hash(args.output)}")


if __name__ == "__main__":
    main()
