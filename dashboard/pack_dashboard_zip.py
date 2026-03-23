#!/usr/bin/env python3
"""
将 dashboard 源码与 output/BASE 下指标 CSV 打成 dashboard.zip，输出到 REAL-T 根目录。

默认同时打包 datasets/REAL-T/BASE/*_meta.csv（与 data.py 一致，否则解压后无法加载元数据）。

缺少 output/BASE、部分指标 CSV 或 meta CSV 时仅打印警告到 stderr，仍会写出 zip（仅打包实际存在的文件）。

运行方式（在 REAL-T 仓库根目录执行，与 Streamlit 一致）::

    cd /path/to/REAL-T
    python3 dashboard/pack_dashboard_zip.py

默认生成同目录下的 ``dashboard.zip``。常用参数::

    python3 dashboard/pack_dashboard_zip.py -o /path/to/out.zip   # 指定输出路径
    python3 dashboard/pack_dashboard_zip.py --dry-run             # 只列出将打包的路径，不写 zip
    python3 dashboard/pack_dashboard_zip.py --no-metadata           # 不打包 datasets/REAL-T/BASE/*_meta.csv

也可直接 ``python3 /path/to/REAL-T/dashboard/pack_dashboard_zip.py``；脚本通过 ``__file__`` 定位仓库根目录。
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


# 与 dashboard/data.py 中 METRIC_SPECS 的 file_suffix 去重后一致
METRIC_FILE_SUFFIXES: tuple[str, ...] = (
    "_TER.csv",
    "_TER_ASR2_AED.csv",
    "_spk_similarity_mixture_enrol.csv",
    "_spk_similarity.csv",
    "_dnsmos.csv",
    "_TSE_TIMING.csv",
)

DASHBOARD_SKIP_DIR_NAMES = frozenset({"__pycache__", ".git", ".mypy_cache", ".ruff_cache"})
DASHBOARD_SKIP_SUFFIXES = frozenset({".pyc", ".pyo"})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def iter_dashboard_files(dashboard_dir: Path) -> list[Path]:
    out: list[Path] = []
    for path in dashboard_dir.rglob("*"):
        if path.is_dir():
            if path.name in DASHBOARD_SKIP_DIR_NAMES:
                continue
            continue
        if any(part in DASHBOARD_SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix in DASHBOARD_SKIP_SUFFIXES:
            continue
        out.append(path)
    return sorted(out)


def collect_metric_csvs(output_base: Path) -> tuple[list[Path], list[str]]:
    """收集存在的指标 CSV；缺失路径仅记入 warnings，不中断打包。"""
    warnings: list[str] = []
    found: list[Path] = []
    if not output_base.is_dir():
        warnings.append(f"未找到 output/BASE 目录，跳过指标 CSV: {output_base}")
        return found, warnings

    for model_dir in sorted(output_base.iterdir(), key=lambda p: p.name.lower()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        missing_suffixes: list[str] = []
        for suffix in METRIC_FILE_SUFFIXES:
            csv_path = model_dir / f"{model}{suffix}"
            if csv_path.is_file():
                found.append(csv_path)
            else:
                missing_suffixes.append(suffix)
        if missing_suffixes:
            warnings.append(
                f"模型 {model!r} 缺少指标 CSV ({len(missing_suffixes)}/{len(METRIC_FILE_SUFFIXES)}): "
                + ", ".join(missing_suffixes)
            )
    return found, warnings


def collect_metadata_csvs(metadata_dir: Path) -> tuple[list[Path], list[str]]:
    warnings: list[str] = []
    if not metadata_dir.is_dir():
        warnings.append(f"未找到元数据目录，跳过 *_meta.csv: {metadata_dir}")
        return [], warnings
    paths = sorted(metadata_dir.glob("*_meta.csv"))
    if not paths:
        warnings.append(f"目录中未找到 *_meta.csv: {metadata_dir}")
    return paths, warnings


def main() -> int:
    root = repo_root()
    dashboard_dir = root / "dashboard"
    output_base = root / "output" / "BASE"
    metadata_dir = root / "datasets" / "REAL-T" / "BASE"

    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "dashboard.zip",
        help=f"zip 输出路径（默认: {root / 'dashboard.zip'})",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="不打包 datasets/REAL-T/BASE/*_meta.csv（解压后需自行保留该目录）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将纳入 zip 的文件列表，不写文件",
    )
    args = parser.parse_args()

    if not dashboard_dir.is_dir():
        print(f"错误: 未找到 dashboard 目录: {dashboard_dir}", file=sys.stderr)
        return 1

    to_add: list[tuple[Path, str]] = []
    for path in iter_dashboard_files(dashboard_dir):
        arc = path.relative_to(root).as_posix()
        to_add.append((path, arc))

    if not to_add:
        print("错误: dashboard 目录下没有可打包的文件", file=sys.stderr)
        return 1

    all_warnings: list[str] = []

    metric_paths, metric_warnings = collect_metric_csvs(output_base)
    all_warnings.extend(metric_warnings)
    for path in metric_paths:
        to_add.append((path, path.relative_to(root).as_posix()))

    if not args.no_metadata:
        meta_paths, meta_warnings = collect_metadata_csvs(metadata_dir)
        all_warnings.extend(meta_warnings)
        for path in meta_paths:
            to_add.append((path, path.relative_to(root).as_posix()))

    for msg in all_warnings:
        print(f"警告: {msg}", file=sys.stderr)

    if args.dry_run:
        for _, arc in to_add:
            print(arc)
        print(f"\n共 {len(to_add)} 个文件", file=sys.stderr)
        return 0

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fs_path, arcname in to_add:
            zf.write(fs_path, arcname)

    print(f"已写入: {out_path}（{len(to_add)} 个文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
