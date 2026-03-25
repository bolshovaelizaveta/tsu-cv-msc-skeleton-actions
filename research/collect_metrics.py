import os
import re
import sys
import ast
import json
import argparse
import subprocess
from collections import Counter


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def canonicalize_label(label: str) -> str:
    return label.strip().lower()


def is_video_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTENSIONS


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def collect_dataset(data_dir: str):
    dataset = []
    class_names = []

    for entry in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(class_dir):
            continue

        class_name = canonicalize_label(entry)
        class_names.append(class_name)

        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            if os.path.isfile(fpath) and is_video_file(fname):
                dataset.append(
                    {
                        "video_path": fpath,
                        "ground_truth": class_name,
                        "file_name": fname,
                    }
                )

    return dataset, sorted(set(class_names))


def parse_target_distribution(stdout_text: str) -> dict:
    """
    Parses block like:
    TARGET CLASS DISTRIBUTION:
    hug                                 — 53.62% (126)
    handshake                           — 17.02% (40)
    ...

    Returns dict[class_name] = count
    """
    lines = stdout_text.splitlines()

    in_block = False
    distribution = {}

    # class name + em dash/hyphen + percent + (count)
    pattern = re.compile(r"^(.*?)\s+[—\-]\s+[\d\.]+%\s+\((\d+)\)\s*$")

    for line in lines:
        stripped = line.strip()

        if "TARGET CLASS DISTRIBUTION:" in stripped:
            in_block = True
            continue

        if in_block:
            if stripped.startswith("========================"):
                break

            if not stripped:
                continue

            match = pattern.match(stripped)
            if match:
                cls_name = canonicalize_label(match.group(1))
                count = int(match.group(2))

                # normalize common printed suffix
                cls_name = cls_name.replace(" (vlm_trigger)", "")
                distribution[cls_name] = count

    return distribution


def parse_final_scene_class(stdout_text: str) -> str | None:
    match = re.search(r"FINAL SCENE CLASS:\s*(.+)", stdout_text)
    if not match:
        return None
    return canonicalize_label(match.group(1))


def parse_scores(stdout_text: str) -> dict:
    match = re.search(r"Scores:\s*(\{.*\})", stdout_text)
    if not match:
        return {}

    raw = match.group(1)
    try:
        return ast.literal_eval(raw)
    except Exception:
        return {}


def run_infer_script(video_path: str, infer_script: str, python_executable: str) -> dict:
    env = os.environ.copy()

    # preserve existing PYTHONPATH, but ensure project root is available
    project_root = os.getcwd()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = project_root

    cmd = [python_executable, infer_script, video_path]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""

    target_distribution = parse_target_distribution(stdout_text)
    final_scene_class = parse_final_scene_class(stdout_text)
    scores = parse_scores(stdout_text)

    top_target_class = "unknown"
    if target_distribution:
        top_target_class = Counter(target_distribution).most_common(1)[0][0]

    error = None
    if proc.returncode != 0:
        error = (
            f"infer script exited with code {proc.returncode}\n"
            f"STDERR:\n{stderr_text}\n"
            f"STDOUT:\n{stdout_text}"
        )

    if final_scene_class is None:
        final_scene_class = "unknown"

    return {
        "returncode": proc.returncode,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "target_distribution": target_distribution,
        "top_target_class": top_target_class,
        "final_scene_class": final_scene_class,
        "scores": scores,
        "error": error,
    }


def compute_binary_metrics_for_class(y_true, y_pred, positive_class: str):
    tp = fp = tn = fn = 0

    for gt, pred in zip(y_true, y_pred):
        gt_pos = gt == positive_class
        pred_pos = pred == positive_class

        if gt_pos and pred_pos:
            tp += 1
        elif not gt_pos and pred_pos:
            fp += 1
        elif gt_pos and not pred_pos:
            fn += 1
        else:
            tn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def compute_macro_micro_metrics(y_true, y_pred, class_names):
    per_class = {}
    for cls in class_names:
        per_class[cls] = compute_binary_metrics_for_class(y_true, y_pred, cls)

    macro_precision = safe_div(sum(m["precision"] for m in per_class.values()), len(class_names))
    macro_recall = safe_div(sum(m["recall"] for m in per_class.values()), len(class_names))
    macro_f1 = safe_div(sum(m["f1"] for m in per_class.values()), len(class_names))
    macro_accuracy = safe_div(sum(m["accuracy"] for m in per_class.values()), len(class_names))

    tp_sum = sum(m["tp"] for m in per_class.values())
    fp_sum = sum(m["fp"] for m in per_class.values())
    fn_sum = sum(m["fn"] for m in per_class.values())

    micro_precision = safe_div(tp_sum, tp_sum + fp_sum)
    micro_recall = safe_div(tp_sum, tp_sum + fn_sum)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    multiclass_accuracy = safe_div(
        sum(int(gt == pred) for gt, pred in zip(y_true, y_pred)),
        len(y_true),
    )

    return {
        "per_class": per_class,
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "accuracy": macro_accuracy,
        },
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
        "multiclass_accuracy": multiclass_accuracy,
    }


def print_metrics(title: str, metrics: dict, class_names: list[str]):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("\nPer-class binary metrics:")
    for cls in class_names:
        m = metrics["per_class"][cls]
        print(
            f"{cls:15s} | "
            f"P={m['precision']:.4f} "
            f"R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} "
            f"Acc={m['accuracy']:.4f} "
            f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}"
        )

    print("\nOverall:")
    print(f"Multiclass Accuracy: {metrics['multiclass_accuracy']:.4f}")
    print(
        f"Macro Precision: {metrics['macro']['precision']:.4f} | "
        f"Macro Recall: {metrics['macro']['recall']:.4f} | "
        f"Macro F1: {metrics['macro']['f1']:.4f} | "
        f"Macro Accuracy: {metrics['macro']['accuracy']:.4f}"
    )
    print(
        f"Micro Precision: {metrics['micro']['precision']:.4f} | "
        f"Micro Recall: {metrics['micro']['recall']:.4f} | "
        f"Micro F1: {metrics['micro']['f1']:.4f}"
    )


def save_csv(video_rows: list[dict], output_csv: str):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fields = [
        "video_path",
        "file_name",
        "ground_truth",
        "top_target_class",
        "final_scene_class",
        "correct_top_target",
        "correct_final",
        "error",
    ]

    with open(output_csv, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for row in video_rows:
            vals = []
            for field in fields:
                val = str(row.get(field, ""))
                val = val.replace('"', '""')
                if "," in val or '"' in val or "\n" in val:
                    val = f'"{val}"'
                vals.append(val)
            f.write(",".join(vals) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--infer-script", default="scripts/infer_stgcnpp_example.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-json", default="results/eval_via_infer_script.json")
    parser.add_argument("--output-csv", default="results/eval_via_infer_script.csv")
    parser.add_argument("--save-stdout", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    dataset, class_names = collect_dataset(args.data_dir)
    if not dataset:
        print(f"No videos found in {args.data_dir}")
        return

    video_rows = []

    y_true = []
    y_pred_final = []
    y_pred_top_target = []

    for idx, item in enumerate(dataset, 1):
        video_path = item["video_path"]
        gt = item["ground_truth"]

        print(f"[{idx}/{len(dataset)}] {video_path}")

        result = run_infer_script(
            video_path=video_path,
            infer_script=args.infer_script,
            python_executable=args.python,
        )

        final_scene_class = canonicalize_label(result["final_scene_class"])
        top_target_class = canonicalize_label(result["top_target_class"])

        correct_final = gt == final_scene_class
        correct_top_target = gt == top_target_class

        print(
            f"  -> GT={gt}, "
            f"TOP_TARGET={top_target_class} [{'CORRECT' if correct_top_target else 'WRONG'}], "
            f"FINAL={final_scene_class} [{'CORRECT' if correct_final else 'WRONG'}]"
        )

        if result["error"]:
            print("  -> ERROR while running infer script")

        row = {
            "video_path": video_path,
            "file_name": item["file_name"],
            "ground_truth": gt,
            "top_target_class": top_target_class,
            "final_scene_class": final_scene_class,
            "correct_top_target": correct_top_target,
            "correct_final": correct_final,
            "target_distribution": result["target_distribution"],
            "scores": result["scores"],
            "error": result["error"],
        }

        if args.save_stdout:
            row["stdout"] = result["stdout"]
            row["stderr"] = result["stderr"]

        video_rows.append(row)

        y_true.append(gt)
        y_pred_final.append(final_scene_class)
        y_pred_top_target.append(top_target_class)

        if result["error"] and args.stop_on_error:
            break

    metrics_final = compute_macro_micro_metrics(y_true, y_pred_final, class_names)
    metrics_top_target = compute_macro_micro_metrics(y_true, y_pred_top_target, class_names)

    print_metrics("METRICS FOR FINAL SCENE CLASS", metrics_final, class_names)
    print_metrics("METRICS FOR TOP TARGET CLASS", metrics_top_target, class_names)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    report = {
        "data_dir": args.data_dir,
        "infer_script": args.infer_script,
        "num_videos": len(video_rows),
        "class_names": class_names,
        "video_results": video_rows,
        "metrics_final_scene_class": metrics_final,
        "metrics_top_target_class": metrics_top_target,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    save_csv(video_rows, args.output_csv)

    print("\nSaved JSON report to:", args.output_json)
    print("Saved CSV report to:", args.output_csv)


if __name__ == "__main__":
    main()