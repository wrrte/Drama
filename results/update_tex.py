import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

RESET_TABLE_VALUES = True
BASE_COLUMN = 5
OURS_COLUMN = 6


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
    if not values:
        return np.nan
    values = np.sort(values)
    trim = int(len(values) * 0.25)
    trimmed = values[trim:len(values) - trim]
    return np.mean(trimmed) if len(trimmed) else np.nan


def reset_drama_values(lines):
    in_table = False
    table_found = False
    for index, line in enumerate(lines):
        if r"\begin{tabular}{lrrrrrrrr}" in line:
            table_found = True
            continue
        if table_found and r"\midrule" in line:
            in_table = True
            continue
        if in_table and r"\bottomrule" in line:
            break
        if in_table and "&" in line:
            parts = line.split("&")
            if len(parts) >= 9:
                parts[BASE_COLUMN] = " - "
                parts[OURS_COLUMN] = " - "
                lines[index] = "&".join(parts)
    return lines


def format_drama_values(tex_path):
    with open(tex_path, encoding="utf-8") as tex_file:
        lines = tex_file.readlines()

    def number_text(cell):
        match = re.search(r"-?\d+\.?\d*", cell)
        return match.group(0) if match else None

    formatted_lines = []
    in_table = False
    table_found = False
    for line in lines:
        if r"\begin{tabular}{lrrrrrrrr}" in line:
            table_found = True
        elif table_found and r"\midrule" in line:
            in_table = True
        elif in_table and r"\bottomrule" in line:
            in_table = False

        if in_table and "&" in line:
            parts = line.split("&")
            if len(parts) >= 9:
                row_label = parts[0].strip()
                base_text = number_text(parts[BASE_COLUMN])
                ours_text = number_text(parts[OURS_COLUMN])
                if base_text is not None and ours_text is not None:
                    base_value = float(base_text)
                    ours_value = float(ours_text)
                    if base_value == 0:
                        difference = (
                            float("inf") if ours_value > 0
                            else float("-inf") if ours_value < 0
                            else 0.0
                        )
                    else:
                        difference = (ours_value - base_value) / abs(base_value) * 100

                    lower_is_better = "Optimality Gap" in row_label
                    improves = difference < 0 if lower_is_better else difference > 0
                    color = "blue" if improves else "red" if difference else "black"
                    if color == "black":
                        formatted = ours_text
                    elif abs(difference) >= 15.0:
                        formatted = f"\\textcolor{{{color}}}{{\\textbf{{{ours_text}}}}}"
                    else:
                        formatted = f"\\textcolor{{{color}}}{{{ours_text}}}"
                    parts[OURS_COLUMN] = f" {formatted} "
                    if OURS_COLUMN == len(parts) - 1:
                        parts[OURS_COLUMN] += r" \\" + "\n"
                    line = "&".join(parts)
        formatted_lines.append(line)

    with open(tex_path, "w", encoding="utf-8") as tex_file:
        tex_file.writelines(formatted_lines)


def load_results(excel_path):
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
            baseline_mean = np.mean([parse_val(baseline_row[seed]) for seed in common_seeds])
            ours_mean = np.mean([parse_val(ours_row[seed]) for seed in common_seeds])
            results[str(game)] = (format_val(baseline_mean), format_val(ours_mean))
            print(
                f"[{game}] Common seeds: {common_seeds} -> "
                f"DRAMA: {results[str(game)][0]}, DRAMA+ours: {results[str(game)][1]}"
            )
    return results


def is_metric_row(line):
    labels = (r"\#Superhuman", "Mean", "Median", "IQM", "Optimality Gap")
    return line.strip().startswith(labels)


def main():
    parser = argparse.ArgumentParser(description="Update DRAMA columns in the main performance table.")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--excel", default=script_dir / "drama_results.xlsx")
    parser.add_argument("--tex", default=script_dir.parent.parent / "iclr2027_conference.tex")
    args = parser.parse_args()

    excel_path = Path(args.excel)
    tex_path = Path(args.tex)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    if not tex_path.exists():
        raise FileNotFoundError(f"TeX file not found: {tex_path}")

    results = load_results(excel_path)
    with open(tex_path, encoding="utf-8") as tex_file:
        original_lines = tex_file.readlines()
    lines = reset_drama_values(original_lines.copy()) if RESET_TABLE_VALUES else original_lines.copy()

    hns_values = {BASE_COLUMN: [], OURS_COLUMN: []}
    for index, original_line in enumerate(original_lines):
        if is_metric_row(original_line):
            continue
        match = re.match(r"^([A-Za-z]+)\s*&", original_line)
        if not match or match.group(1) == "Game":
            continue

        parts = original_line.split("&")
        if len(parts) < 9:
            continue
        game = match.group(1)
        output_parts = lines[index].split("&")
        if game in results:
            output_parts[BASE_COLUMN] = f" {results[game][0]} "
            output_parts[OURS_COLUMN] = f" {results[game][1]} "
        lines[index] = "&".join(output_parts)

        random_value = extract_float(parts[1])
        human_value = extract_float(parts[2])
        if random_value is None or human_value is None or human_value == random_value:
            continue
        for column in (BASE_COLUMN, OURS_COLUMN):
            score = extract_float(parts[column])
            if score is not None:
                hns_values[column].append((score - random_value) / (human_value - random_value))

    metrics = {"#Superhuman": {}, "Mean": {}, "Median": {}, "IQM": {}, "Optimality Gap": {}}
    for column, values in hns_values.items():
        metrics["#Superhuman"][column] = sum(value > 1.0 for value in values)
        metrics["Mean"][column] = np.mean(values) if values else np.nan
        metrics["Median"][column] = np.median(values) if values else np.nan
        metrics["IQM"][column] = calc_iqm(values)
        metrics["Optimality Gap"][column] = (
            np.mean([max(0.0, 1.0 - value) for value in values]) if values else np.nan
        )

    for index, line in enumerate(lines):
        metric = next(
            (
                name for name in metrics
                if line.strip().startswith(name if name != "#Superhuman" else r"\#Superhuman")
            ),
            None,
        )
        if metric is None:
            continue
        parts = line.split("&")
        if len(parts) < 9:
            continue
        for column in (BASE_COLUMN, OURS_COLUMN):
            value = metrics[metric][column]
            formatted = " - " if np.isnan(value) else (
                f" {int(value)} " if metric == "#Superhuman" else f" {value:.3f} "
            )
            parts[column] = formatted
        lines[index] = "&".join(parts)

    with open(tex_path, "w", encoding="utf-8") as tex_file:
        tex_file.writelines(lines)
    format_drama_values(tex_path)
    print(f"Successfully updated {tex_path} with DRAMA and DRAMA+ours results.")


if __name__ == "__main__":
    main()
