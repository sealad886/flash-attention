"""Utilities for precompiling MLX Metal kernels."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Iterable, Optional, Sequence

from .kernels import list_registered_kernels, precompile_kernels

_VARLEN_KERNEL_NAMES: Sequence[str] = (
    "flash_attention_varlen_fwd",
    "flash_attention_varlen_bwd_dq",
    "flash_attention_varlen_bwd_dkv",
)


def precompile_varlen_kernels(*, skip_missing_fast: bool = True) -> Dict[str, Any]:
    """Precompile the varlen Flash Attention kernels."""

    return precompile_kernels(_VARLEN_KERNEL_NAMES, skip_missing_fast=skip_missing_fast)


def precompile_all_kernels(*, skip_missing_fast: bool = True) -> Dict[str, Any]:
    """Precompile every registered MLX kernel."""

    return precompile_kernels(skip_missing_fast=skip_missing_fast)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the registered kernel names and exit.",
    )
    parser.add_argument(
        "--kernels",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Explicit kernel names to precompile. Default: all registered.",
    )
    parser.add_argument(
        "--varlen-only",
        action="store_true",
        help="Precompile only the varlen Flash Attention kernels.",
    )
    parser.add_argument(
        "--skip-missing-fast",
        action="store_true",
        default=False,
        help=(
            "If set, return success even when mlx.fast is unavailable instead of "
            "raising a RuntimeError."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list:
        for name in list_registered_kernels():
            print(name)
        return 0

    if args.varlen_only and args.kernels:
        raise ValueError("--varlen-only cannot be combined with explicit --kernels")

    if args.varlen_only:
        targets: Optional[Iterable[str]] = _VARLEN_KERNEL_NAMES
    else:
        targets = args.kernels

    compiled = precompile_kernels(targets, skip_missing_fast=args.skip_missing_fast)
    print(f"Precompiled {len(compiled)} kernel(s): {', '.join(compiled.keys())}")
    return 0


def entrypoint() -> None:
    """Console entry point used by ``python -m flash_attn.flash_attn_mlx.precompile``."""

    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    entrypoint()
