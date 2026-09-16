import argparse
import re
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

# Reset only this method's score and delta cells before updating.
RESET_TABLE_VALUES = True
BASE_COLUMN = 6
OURS_COLUMN = 7
DELTA_COLUMN = 8


def parse_val(value):
    value = str(value).strip()
    if value.lower() in {"", "nan", "n/a", "na", "none", "running"}:
        return np.nan
    value = value.split(",")[0].split("(")[0].strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def format_val(value):
    if np.isnan(value):
        return "-"
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def extract_float(value):
    match = re.search(r"-?\d+\.?\d*", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def calc_iqm(values):
    """Trim 25% from each tail of the pooled game-by-seed HNS values."""
    if not values:
        return np.nan
    values = np.sort(values)
    trim = int(len(values) * 0.25)
    trimmed = values[trim:len(values) - trim]
    return np.mean(trimmed) if len(trimmed) else np.nan


def main_table_rows(lines):
    """Return only data rows belonging to tab:main_performance."""
    label_index = next(
        (index for index, line in enumerate(lines) if r"\label{tab:main_performance}" in line
         and not line.lstrip().startswith("%")),
        None,
    )
    if label_index is None:
        raise ValueError("Could not find tab:main_performance.")

    expected_header = [
        "Game", "Random", "Human", "STORM", "STORM+ours", r"$\Delta$",
        "DRAMA", "DRAMA+ours", r"$\Delta$",
    ]
    rows = []
    header_found = False
    for index in range(label_index + 1, len(lines)):
        line = lines[index]
        if line.lstrip().startswith("%"):
            continue
        if any(marker in line for marker in (r"\bottomrule", r"\end{tabular}", r"\end{table}")):
            if header_found:
                return rows
            break
        if "&" not in line:
            continue
        body, separator, tail = line.partition(r"\\")
        parts = body.split("&")
        if parts[0].strip() == "Game":
            if [part.strip() for part in parts] != expected_header:
                raise ValueError("Unexpected column layout in tab:main_performance.")
            header_found = True
        elif header_found:
            if len(parts) != len(expected_header) or not separator:
                raise ValueError(f"Malformed main performance row at line {index + 1}.")
            rows.append((index, parts, separator + tail))

    raise ValueError("Could not find the main performance table header and end.")


def number_text(cell):
    # Remove color specifications before extracting numbers (e.g. green!50!black).
    cell = re.sub(r"\\textcolor\{[^{}]*\}", "", cell)
    match = re.search(r"[+-]?\d+(?:\.\d+)?", cell.replace(",", ""))
    return match.group(0) if match else None


def metric_name(label):
    label = label.strip().lstrip("\\")
    return next(
        (name for name in ("#Superhuman", "Mean", "Median", "IQM", "Optimality Gap")
         if label.startswith(name)),
        None,
    )


def format_delta(base_text, ours_text, metric):
    if base_text is None or ours_text is None:
        return "-"
    # Subtract displayed decimal values exactly so delta agrees with the table.
    delta = Decimal(ours_text) - Decimal(base_text)
    if metric and metric != "#Superhuman":
        text = f"{delta:.3f}"
    else:
        text = format(delta, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
    if delta == 0:
        return text.lstrip("-")
    if delta > 0:
        text = "+" + text
    # Lower Optimality Gap is better; other metrics improve when they increase.
    improves = delta < 0 if metric == "Optimality Gap" else delta > 0
    color = "green" if improves else "red"
    return f"\\textcolor{{{color}}}{{{text}}}"


def reset_values(lines):
    """Clear only this method's baseline, +ours, and delta cells."""
    for index, parts, ending in main_table_rows(lines):
        for column in (BASE_COLUMN, OURS_COLUMN, DELTA_COLUMN):
            parts[column] = " - "
        lines[index] = "&".join(parts) + ending
    return lines


def format_values(lines):
    for index, parts, ending in main_table_rows(lines):
        base_text = number_text(parts[BASE_COLUMN])
        ours_text = number_text(parts[OURS_COLUMN])
        parts[OURS_COLUMN] = f" {ours_text if ours_text is not None else '-'} "
        delta = format_delta(base_text, ours_text, metric_name(parts[0]))
        parts[DELTA_COLUMN] = f" {delta} "
        lines[index] = "&".join(parts) + ending
    return lines


def update_table(lines, results, reset=True):
    """Render game means, using unrounded per-seed results only for IQM."""
    lines = reset_values(lines.copy()) if reset else lines.copy()
    hns_values = {BASE_COLUMN: [], OURS_COLUMN: []}
    seed_hns_values = {BASE_COLUMN: [], OURS_COLUMN: []}
    for index, parts, ending in main_table_rows(lines):
        if metric_name(parts[0]):
            continue
        game = parts[0].strip()
        if game in results:
            for column, scores in zip((BASE_COLUMN, OURS_COLUMN), results[game]):
                parts[column] = f" {format_val(np.mean(scores))} "
        lines[index] = "&".join(parts) + ending

        random_value = extract_float(parts[1])
        human_value = extract_float(parts[2])
        if random_value is None or human_value is None or human_value == random_value:
            continue
        if game in results:
            # Each matched (game, training seed) contributes one score. Normalize
            # before pooling; never recover IQM inputs from rounded game means.
            for column, scores in zip((BASE_COLUMN, OURS_COLUMN), results[game]):
                seed_hns_values[column].extend(
                    (score - random_value) / (human_value - random_value)
                    for score in scores
                )
        # Use the updated scores, never the previous table or delta columns.
        for column in (BASE_COLUMN, OURS_COLUMN):
            score_text = number_text(parts[column])
            if score_text is not None:
                hns_values[column].append(
                    (float(score_text) - random_value) / (human_value - random_value)
                )

    metrics = {"#Superhuman": {}, "Mean": {}, "Median": {}, "IQM": {}, "Optimality Gap": {}}
    for column, values in hns_values.items():
        metrics["#Superhuman"][column] = sum(value > 1.0 for value in values) if values else np.nan
        metrics["Mean"][column] = np.mean(values) if values else np.nan
        metrics["Median"][column] = np.median(values) if values else np.nan
        metrics["IQM"][column] = calc_iqm(seed_hns_values[column])
        metrics["Optimality Gap"][column] = (
            np.mean([max(0.0, 1.0 - value) for value in values]) if values else np.nan
        )

    for index, parts, ending in main_table_rows(lines):
        metric = metric_name(parts[0])
        if metric is None:
            continue
        for column in (BASE_COLUMN, OURS_COLUMN):
            value = metrics[metric][column]
            formatted = "-" if np.isnan(value) else (
                str(int(value)) if metric == "#Superhuman" else f"{value:.3f}"
            )
            parts[column] = f" {formatted} "
        lines[index] = "&".join(parts) + ending

    return format_values(lines)


def reset_drama_values(lines):
    return reset_values(lines)


def format_drama_values(tex_path):
    with open(tex_path, encoding="utf-8") as tex_file:
        lines = format_values(tex_file.readlines())
    with open(tex_path, "w", encoding="utf-8") as tex_file:
        tex_file.writelines(lines)


def load_results(excel_path):
    """Keep both methods' raw scores for the same valid training seeds."""
    frame = pd.read_excel(excel_path, sheet_name="Results")
    required = {"Game", "Retrieval"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"Results sheet is missing columns: {missing}")

    seed_columns = [column for column in frame.columns if str(column).strip().isdigit()]
    results = {}
    for game in frame["Game"].dropna().unique():
        game_frame = frame[frame["Game"] == game]
        # Drama uses X for retrieval-off and O for retrieval-on.
        baseline = game_frame[game_frame["Retrieval"].astype(str).str.upper() == "X"]
        ours = game_frame[game_frame["Retrieval"].astype(str).str.upper() == "O"]
        if baseline.empty or ours.empty:
            continue

        baseline_row = baseline.iloc[0]
        ours_row = ours.iloc[0]
        common_seeds = [
            seed for seed in seed_columns
            if not np.isnan(parse_val(baseline_row[seed]))
            and not np.isnan(parse_val(ours_row[seed]))
        ]
        if common_seeds:
            baseline_scores = [parse_val(baseline_row[seed]) for seed in common_seeds]
            ours_scores = [parse_val(ours_row[seed]) for seed in common_seeds]
            results[str(game)] = (baseline_scores, ours_scores)
            print(
                f"[{game}] Common seeds: {common_seeds} -> "
                f"DRAMA: {format_val(np.mean(baseline_scores))}, "
                f"DRAMA+ours: {format_val(np.mean(ours_scores))}"
            )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Update DRAMA scores and deltas in the main performance table."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--excel", type=Path, default=script_dir / "drama_results.xlsx")
    parser.add_argument("--tex", type=Path, default=script_dir.parent.parent / "iclr2027_conference.tex")
    args = parser.parse_args()

    if not args.excel.exists():
        raise FileNotFoundError(f"Excel file not found: {args.excel}")
    if not args.tex.exists():
        raise FileNotFoundError(f"TeX file not found: {args.tex}")

    results = load_results(args.excel)
    with args.tex.open(encoding="utf-8") as tex_file:
        lines = tex_file.readlines()
    lines = update_table(lines, results, reset=RESET_TABLE_VALUES)
    with args.tex.open("w", encoding="utf-8") as tex_file:
        tex_file.writelines(lines)
    print(f"Successfully updated {args.tex} with DRAMA scores, metrics, and deltas.")


if __name__ == "__main__":
    main()
