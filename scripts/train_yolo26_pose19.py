import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_DATASET_ROOT = r"F:\datasets\coco8-pose\coco8-pose"
DEFAULT_DATA_YAML = "configs/coco8_pose19.yaml"
DEFAULT_MODEL = "data/models/yolo26n-pose.pt"


def iter_label_files(label_root: Path) -> Iterable[Path]:
    if not label_root.exists():
        return []
    return sorted(p for p in label_root.rglob("*.txt") if p.is_file())


def write_pose19_yaml(dataset_root: Path, yaml_path: Path) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"path: {dataset_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [19, 3]\n"
        "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]\n"
        "names:\n"
        "  0: person\n"
    )
    yaml_path.write_text(content, encoding="utf-8")


def convert_line_17_to_19(line: str, toe_scale: float) -> str:
    parts = [x for x in line.strip().split() if x]
    if not parts:
        return line
    if len(parts) == 62:
        return line.strip()
    if len(parts) != 56:
        raise ValueError(f"Unexpected label length {len(parts)}, expected 56 or 62")

    values = [float(x) for x in parts]
    kp = values[5:]  # 17 * 3
    kps = [kp[i * 3 : i * 3 + 3] for i in range(17)]

    def build_toe(knee_idx: int, ankle_idx: int) -> List[float]:
        knee_x, knee_y, knee_v = kps[knee_idx]
        ankle_x, ankle_y, ankle_v = kps[ankle_idx]
        if ankle_v <= 0:
            return [0.0, 0.0, 0.0]
        if knee_v > 0:
            toe_x = ankle_x + toe_scale * (ankle_x - knee_x)
            toe_y = ankle_y + toe_scale * (ankle_y - knee_y)
            toe_x = min(max(toe_x, 0.0), 1.0)
            toe_y = min(max(toe_y, 0.0), 1.0)
            return [toe_x, toe_y, ankle_v]
        return [ankle_x, ankle_y, ankle_v]

    left_toe = build_toe(13, 15)
    right_toe = build_toe(14, 16)
    values.extend(left_toe)
    values.extend(right_toe)
    return " ".join(f"{v:.6f}" for v in values)


def bootstrap_labels_17_to_19(dataset_root: Path, backup_root: Path, toe_scale: float) -> Tuple[int, int]:
    label_root = dataset_root / "labels"
    files = list(iter_label_files(label_root))
    if not files:
        raise FileNotFoundError(f"No label files found under {label_root}")

    if backup_root.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_root}")
    shutil.copytree(label_root, backup_root)

    converted_files = 0
    converted_lines = 0
    for file_path in files:
        raw_lines = file_path.read_text(encoding="utf-8").splitlines()
        new_lines: List[str] = []
        changed = False
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                new_lines.append(line)
                continue
            converted = convert_line_17_to_19(stripped, toe_scale)
            if converted != stripped:
                changed = True
                converted_lines += 1
            new_lines.append(converted)
        if changed:
            converted_files += 1
            file_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return converted_files, converted_lines


def validate_pose19_labels(dataset_root: Path) -> Tuple[int, int]:
    label_root = dataset_root / "labels"
    files = list(iter_label_files(label_root))
    if not files:
        raise FileNotFoundError(f"No label files found under {label_root}")

    bad_lines = 0
    checked_lines = 0
    for file_path in files:
        for line_no, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            checked_lines += 1
            n = len(stripped.split())
            if n != 62:  # cls + bbox(4) + 19*3
                bad_lines += 1
                if bad_lines <= 20:
                    print(f"[BAD] {file_path}:{line_no} has {n} values (expect 62)")
    return checked_lines, bad_lines


def train(args: argparse.Namespace, data_yaml_path: Path) -> None:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "ultralytics is not installed in current env. Install it first: pip install ultralytics"
        ) from exc

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Put weights in data/models or pass --model explicitly."
        )

    model = YOLO(str(model_path))
    model.train(
        task="pose",
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=True,
        exist_ok=args.exist_ok,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO26 pose model to 19 keypoints (17 + left/right toe)."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Dataset root path")
    parser.add_argument("--data-yaml", default=DEFAULT_DATA_YAML, help="Output data yaml path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model, e.g. yolo26n-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU id, cpu, or comma list")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/pose19")
    parser.add_argument("--name", default="yolo26n_pose19_ft")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--bootstrap-toe-from-ankle",
        action="store_true",
        help="Convert 17-kpt labels to 19-kpt in place by bootstrapping toe points from knee-ankle geometry.",
    )
    parser.add_argument(
        "--toe-bootstrap-scale",
        type=float,
        default=0.45,
        help="Toe bootstrap scale: toe = ankle + scale * (ankle - knee).",
    )
    parser.add_argument(
        "--backup-dir",
        default="",
        help="Backup directory for original labels before bootstrap conversion. Default: <dataset-root>/labels_backup_17kpt",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only prepare and validate, do not train")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    data_yaml_path = Path(args.data_yaml)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    write_pose19_yaml(dataset_root, data_yaml_path)
    print(f"[OK] Pose19 data yaml written: {data_yaml_path}")

    if args.bootstrap_toe_from_ankle:
        backup_dir = Path(args.backup_dir) if args.backup_dir else (dataset_root / "labels_backup_17kpt")
        converted_files, converted_lines = bootstrap_labels_17_to_19(
            dataset_root, backup_dir, args.toe_bootstrap_scale
        )
        print(f"[OK] Bootstrap done. backup={backup_dir}")
        print(f"[OK] Converted files={converted_files}, lines={converted_lines}")
        print("[WARN] This is temporary pseudo-labeling. Real toe labels are still recommended.")

    checked, bad = validate_pose19_labels(dataset_root)
    print(f"[OK] Checked label lines={checked}")
    if bad > 0:
        raise RuntimeError(
            f"Found {bad} invalid lines. For pose19, each non-empty line must have 62 values."
        )

    if args.dry_run:
        print("[OK] Dry run finished.")
        return

    train(args, data_yaml_path)


if __name__ == "__main__":
    main()
