#!/usr/bin/env python3

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd):
    print(f"[RUN] {' '.join(shlex.quote(x) for x in cmd)}")
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def find_models(weights_dir: Path):
    return sorted(weights_dir.glob("*/model_best_bp2_serialize.pth"))


def target_dir_name(model_path: Path, valid_iters: int) -> str:
    model_group = model_path.parent.name
    return f"onnx_{model_group}_iter{valid_iters}"


def is_target_complete(target_dir: Path) -> bool:
    required = [
        target_dir / "feature_runner.engine",
        target_dir / "post_runner.engine",
        target_dir / "feature_runner.onnx",
        target_dir / "post_runner.onnx",
        target_dir / "onnx.yaml",
    ]
    return all(p.is_file() for p in required)


def build_one(project_root: Path, model_path: Path, valid_iters: int, height: int, width: int, max_disp: int):
    out_dir = project_root / target_dir_name(model_path, valid_iters)
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_target_complete(out_dir):
        print(f"[SKIP] Already complete: {out_dir}")
        return "skipped"

    make_onnx_cmd = [
        "python3",
        "make_onnx.py",
        "--model_dir",
        str(model_path),
        "--save_path",
        str(out_dir),
        "--height",
        str(height),
        "--width",
        str(width),
        "--valid_iters",
        str(valid_iters),
        "--max_disp",
        str(max_disp),
    ]
    run_cmd(make_onnx_cmd, cwd=project_root)

    feature_onnx = out_dir / "feature_runner.onnx"
    post_onnx = out_dir / "post_runner.onnx"

    if not feature_onnx.is_file() or not post_onnx.is_file():
        raise RuntimeError(f"ONNX export missing in {out_dir}")

    if not (out_dir / "feature_runner.engine").is_file():
        run_cmd(
            [
                "trtexec",
                f"--onnx={feature_onnx}",
                f"--saveEngine={out_dir / 'feature_runner.engine'}",
                "--fp16",
                "--useCudaGraph",
            ],
            cwd=project_root,
        )

    if not (out_dir / "post_runner.engine").is_file():
        run_cmd(
            [
                "trtexec",
                f"--onnx={post_onnx}",
                f"--saveEngine={out_dir / 'post_runner.engine'}",
                "--fp16",
                "--useCudaGraph",
            ],
            cwd=project_root,
        )

    if not is_target_complete(out_dir):
        raise RuntimeError(f"Target did not complete successfully: {out_dir}")

    print(f"[DONE] Built: {out_dir}")
    return "built"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build ONNX + TensorRT engines for each weights/*/model_best_bp2_serialize.pth "
            "with valid_iters in {4,8}. Skips already-complete targets."
        )
    )
    parser.add_argument("--project-root", default=".", help="Path to foundation_stereo_ros root")
    parser.add_argument("--weights-dir", default="weights", help="Relative weights directory")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--iters", nargs="+", type=int, default=[4, 8], help="valid_iters values to build")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    weights_dir = (project_root / args.weights_dir).resolve()

    if not (project_root / "make_onnx.py").is_file():
        raise RuntimeError(f"make_onnx.py not found in project root: {project_root}")

    if not weights_dir.is_dir():
        raise RuntimeError(f"weights dir not found: {weights_dir}")

    if shutil_which("trtexec") is None:
        raise RuntimeError("trtexec not found in PATH. Install TensorRT CLI first.")

    models = find_models(weights_dir)
    if not models:
        print("No models found under weights/*/model_best_bp2_serialize.pth")
        return 0

    plan = []
    for model_path in models:
        for it in args.iters:
            plan.append((model_path, it, project_root / target_dir_name(model_path, it)))

    print("Build plan:")
    for model_path, it, out_dir in plan:
        status = "SKIP" if is_target_complete(out_dir) else "BUILD"
        print(f"  - [{status}] {model_path} -> {out_dir} (valid_iters={it})")

    if args.dry_run:
        return 0

    built = 0
    skipped = 0
    for model_path, it, _ in plan:
        result = build_one(
            project_root=project_root,
            model_path=model_path,
            valid_iters=it,
            height=args.height,
            width=args.width,
            max_disp=args.max_disp,
        )
        if result == "built":
            built += 1
        else:
            skipped += 1

    print(f"Summary: built={built}, skipped={skipped}, total={len(plan)}")
    return 0


def shutil_which(cmd: str):
    from shutil import which

    return which(cmd)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
