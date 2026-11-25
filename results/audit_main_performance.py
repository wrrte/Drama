"""Reproduce the DRAMA columns of tab:main_performance without editing inputs.

Run with the storm conda environment (numpy, pandas, openpyxl). The frozen
source file includes run IDs and the six scores missing from the old workbook.
All computation and verification here are offline.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

import update_tex as updater


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sources(records, workbook, csv_path):
    """Check selected Excel cells and CSV run IDs, including recorded metadata."""
    book = load_workbook(workbook, data_only=True)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = {row["Run ID"]: row for row in csv.DictReader(stream)}
    keys = [(r["game"], r["seed"], r["mode"]) for r in records]
    require(len(keys) == len(set(keys)), "Duplicate game/seed/mode in frozen sources")
    for row in records:
        label = f"{row['game']} seed {row['seed']} {row['mode']}"
        config = row["retrieval"]
        require(
            (config["warmup_steps"], config["anchor_offset"],
             config["anchor_weight"], config["target"])
            == (50000, -2, 0.0625, 16), f"Unexpected retrieval settings: {label}",
        )
        require(config["enable"] == (row["mode"] == "O"), f"Wrong mode: {label}")
        if row["wandb_eval_score"] is not None:
            require(row["score"] == row["wandb_eval_score"], f"W&B score differs: {label}")
        else:
            require(row["workbook_cell"] is not None, f"No score source: {label}")
        if not row["workbook_cell"]:
            continue
        sheet, address = row["workbook_cell"].split("!")
        require(updater.parse_val(book[sheet][address].value) == row["score"],
                f"Workbook score changed: {label}")
        run_sheet = book["Runs"]
        headers = {cell.value: cell.column for cell in run_sheet[1]}
        run_row = row["workbook_runs_row"]
        for column, expected in [("Game", row["game"]), ("Seed", row["seed"]),
                                 ("Retrieval", row["mode"]), ("Run ID", row["run_id"])]:
            require(run_sheet.cell(run_row, headers[column]).value == expected,
                    f"Workbook run metadata changed: {label}")
        require(updater.parse_val(run_sheet.cell(run_row, headers["Eval Return"]).value)
                == row["score"], f"Runs score changed: {label}")
        source = csv_rows[row["run_id"]]
        require((source["Game"], int(source["Seed"]), source["Retrieval"])
                == (row["game"], row["seed"], row["mode"]), f"CSV metadata changed: {label}")
        require(float(source["Eval Return"]) == row["score"], f"CSV score changed: {label}")
    book.close()


def audit(manifest, tex_path, workbook):
    records = manifest["records"]
    lines = tex_path.read_text(encoding="utf-8").splitlines(keepends=True)
    table = {parts[0].strip(): parts for _, parts, _ in updater.main_table_rows(lines)}
    games = [name for name in table if updater.metric_name(name) is None]
    require(set(games) == {r["game"] for r in records}, "Game coverage differs from paper")
    results, game_rows = {}, []
    game_hns, seed_hns = {"X": [], "O": []}, {"X": [], "O": []}
    for game in games:
        parts = table[game]
        selected = {mode: sorted((r for r in records if r["game"] == game
                                 and r["mode"] == mode), key=lambda r: r["seed"])
                    for mode in ("X", "O")}
        seeds = [r["seed"] for r in selected["X"]]
        require(seeds == [r["seed"] for r in selected["O"]], f"Unpaired seeds: {game}")
        results[game] = tuple([r["score"] for r in selected[m]] for m in ("X", "O"))
        row = {"Game": game, "Seeds": ", ".join(map(str, seeds)), "N": len(seeds)}
        random_score, human_score = (float(updater.number_text(parts[i])) for i in (1, 2))
        for mode, column, method in [("X", 6, "DRAMA"), ("O", 7, "Full FLASH")]:
            scores = [r["score"] for r in selected[mode]]
            displayed = updater.format_val(np.mean(scores))
            paper = updater.number_text(parts[column])
            row.update({f"{method} raw mean": np.mean(scores), f"{method} reproduced": displayed,
                        f"{method} paper": paper, f"{method} match": Decimal(displayed) == Decimal(paper)})
            game_hns[mode].append((float(displayed) - random_score) / (human_score - random_score))
            seed_hns[mode].extend((x - random_score) / (human_score - random_score) for x in scores)
        row["Delta reproduced"] = str(Decimal(row["Full FLASH reproduced"]) - Decimal(row["DRAMA reproduced"]))
        row["Delta paper"] = updater.number_text(parts[8])
        row["Delta match"] = Decimal(row["Delta reproduced"]) == Decimal(row["Delta paper"])
        game_rows.append(row)

    metric_rows = []
    for name, parts in table.items():
        metric = updater.metric_name(name)
        if metric is None:
            continue
        row = {"Metric": metric}
        for mode, column, method in [("X", 6, "DRAMA"), ("O", 7, "Full FLASH")]:
            values = np.asarray(game_hns[mode])
            pooled = np.sort(seed_hns[mode])
            trim = len(pooled) // 4
            metrics = {"#Superhuman": int(np.sum(values > 1)), "Mean": np.mean(values),
                       "Median": np.median(values), "IQM": np.mean(pooled[trim:len(pooled)-trim]),
                       "Optimality Gap": np.mean(np.maximum(0, 1 - values))}
            value = metrics[metric]
            text = str(value) if metric == "#Superhuman" else f"{value:.3f}"
            paper = updater.number_text(parts[column])
            row.update({f"{method} calculated": value, f"{method} reproduced": text,
                        f"{method} paper": paper, f"{method} match": Decimal(text) == Decimal(paper)})
        row["Delta reproduced"] = str(Decimal(row["Full FLASH reproduced"]) - Decimal(row["DRAMA reproduced"]))
        row["Delta paper"] = updater.number_text(parts[8])
        row["Delta match"] = Decimal(row["Delta reproduced"]) == Decimal(row["Delta paper"])
        metric_rows.append(row)

    # Cross-check independent aggregation against the repository's paper renderer.
    rendered = updater.update_table(lines, results)
    for _, parts, _ in updater.main_table_rows(rendered):
        original = table[parts[0].strip()]
        for column in (6, 7, 8):
            require(Decimal(updater.number_text(parts[column])) == Decimal(updater.number_text(original[column])),
                    f"Paper renderer differs: {parts[0].strip()}, column {column}")

    with contextlib.redirect_stdout(io.StringIO()):
        current = updater.load_results(workbook)
    current_rows = []
    for row in game_rows:
        scores = current.get(row["Game"])
        baseline, flash = ([updater.format_val(np.mean(s)) for s in scores] if scores else [None, None])
        current_rows.append({"Game": row["Game"], "Current workbook DRAMA": baseline,
                             "Current workbook Full FLASH": flash,
                             "Paper DRAMA": row["DRAMA paper"], "Paper Full FLASH": row["Full FLASH paper"],
                             "Match": (baseline, flash) == (row["DRAMA paper"], row["Full FLASH paper"])})
    for row in game_rows + metric_rows:
        require(all(row[k] for k in ("DRAMA match", "Full FLASH match", "Delta match")),
                f"Paper reproduction failed: {row}")
    return game_rows, metric_rows, current_rows


def write_report(manifest, games, metrics, current, output):
    output.mkdir(parents=True, exist_ok=True)
    records = manifest["records"]
    metadata = [{"Item": "Retrieved UTC", "Value": manifest["retrieved_at_utc"]}]
    metadata += [{"Item": f"SHA256 {name}", "Value": info["sha256"]}
                 for name, info in manifest["inputs"].items()]
    metadata += [{"Item": "Method / limitation", "Value": note} for note in manifest["selection_notes"]]
    flat_scores = [{**{k: v for k, v in r.items() if k != "retrieval"},
                    **{f"config.{k}": v for k, v in r["retrieval"].items()}}
                   for r in records]
    with pd.ExcelWriter(output / "main_performance_seeds.xlsx", engine="openpyxl") as writer:
        for name, rows in [("Games", games), ("Selected Scores", flat_scores), ("Metrics", metrics),
                           ("Current Loader", current), ("Freeway Candidates", manifest["freeway_candidates"]),
                           ("Provenance", metadata)]:
            pd.DataFrame(rows).to_excel(writer, sheet_name=name, index=False)
            ws = writer.book[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for cells in ws.columns:
                ws.column_dimensions[cells[0].column_letter].width = min(64, max(12, max(len(str(c.value or "")) for c in cells) + 2))
    pd.DataFrame(flat_scores).to_csv(output / "main_performance_selected_scores.csv", index=False)
    text = ["# DRAMA main performance 게임·시드 대조표", "",
            "기준: `iclr2027_conference.tex`의 `tab:main_performance`, DRAMA / DRAMA+ours(Full FLASH) 열.",
            "26게임의 양쪽 점수, 네 지표, #Superhuman 및 모든 Δ가 논문 표시값과 일치한다.",
            "총 102개의 공통 (게임, 시드)쌍: Freeway는 2쌍, 나머지 25게임은 각각 4쌍이다.", "",
            "| 게임 | 공통 시드 | N | DRAMA | Full FLASH | 일치 |",
            "|---|---|---:|---:|---:|---|"]
    text += [f"| {r['Game']} | {r['Seeds']} | {r['N']} | {r['DRAMA reproduced']} | {r['Full FLASH reproduced']} | O |" for r in games]
    text += ["", "| 지표 | DRAMA 재계산 | 논문 | Full FLASH 재계산 | 논문 | 표시값 일치 |", "|---|---:|---:|---:|---:|---|"]
    text += [f"| {r['Metric']} | {r['DRAMA calculated']:.9f} | {r['DRAMA paper']} | {r['Full FLASH calculated']:.9f} | {r['Full FLASH paper']} | O |" for r in metrics]
    text += ["", "HNS = (score − Random) / (Human − Random). Mean·Median·Optimality Gap은 게임을 동일 가중하며, 기존 논문 집계 규칙대로 표시된 게임 평균을 정규화한다. IQM은 반올림 전 102개 시드별 HNS를 모아 양쪽 꼬리에서 각각 25개를 제거하고 남은 52개를 평균한다.", "",
             "현재 `drama_results.xlsx`만 기존 loader로 읽으면 24게임만 일치한다. Krull은 기존 710 대신 2000을 사용해야 하며, Freeway는 workbook에서 누락되어 W&B 점수를 보완해야 한다.", "",
             "- Krull seed 2000: DRAMA 9036 (`21ijqipq`), Full FLASH 11093 (`mjjum96z`). 선택 시드는 1710·2000·3710·6000이다.",
             "- Freeway seed 2710: DRAMA 31.3 (`vpxa1ak7`), Full FLASH 32.2 (`m3lhpghb`). seed 2030: 양쪽 0 (`1fcphg7b` / `l96lmgvj`). 두 시드 평균은 15.65 / 16.1로 논문 표시에 따라 15.7 / 16.1이다.",
             "- Freeway seed 2040도 양쪽 0이므로 2030 대신 넣으면 같은 점수가 된다. 다만 2040 baseline의 마지막 기록은 2026-09-26 13:07 KST이고, 논문 수정 커밋 `c6342e0`은 같은 날 10:09 KST이다. 이 시간 기록에 근거해 원래 논문에는 2030이 사용된 것으로 판단했다. 두 zero 시드를 모두 넣으면 논문과 달라진다.",
             "- 모든 204개 run의 설정은 warmup_steps=50000, anchor_offset=-2, anchor_weight=0.0625, target=16이며 X/O 모드도 확인했다.",
             "- 204개 중 185개 점수는 조회 당시 W&B `evaluate/score`와 정확히 일치한다. 나머지 19개는 W&B 최종 평가값이 없어 기존 Excel·CSV 기록의 일치까지만 검증했다. 이 중 4개는 `classify_wandb_runs.py`의 수동 보완값이다. 학습이나 평가를 다시 실행한 결과는 아니다.",
             "- 논문의 모든 게임 4시드라는 설명과 Freeway의 실제 2시드는 차이가 있다.", "",
             "`main_performance_sources.json`은 run ID·설정·점수·출처·원본 SHA256을 고정한다. Excel의 `Selected Scores` 시트에는 204개 점수의 원본 셀과 CSV 행, W&B 검증 상태를 기록했다. `Current Loader` 시트는 현재 workbook 집계와의 차이를 보여준다.", "",
             "원본 workbook·CSV와 논문은 수정하지 않았다. 결과는 다음 명령으로 오프라인 재검증할 수 있다.", "", "```bash",
             "conda run -n storm python Drama/results/audit_main_performance.py", "```", "",
             f"W&B 조회 시각(UTC): {manifest['retrieved_at_utc']}", ""]
    (output / "main_performance_seeds.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=HERE / "main_performance_sources.json")
    parser.add_argument("--workbook", type=Path, default=HERE / "drama_results.xlsx")
    parser.add_argument("--csv", type=Path, default=HERE / "drama_wandb_runs.csv")
    parser.add_argument("--tex", type=Path, default=ROOT / "iclr2027_conference.tex")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    before = {p: sha256(p) for p in (args.workbook, args.csv, args.tex)}
    manifest = json.loads(args.sources.read_text(encoding="utf-8"))
    validate_sources(manifest["records"], args.workbook, args.csv)
    games, metrics, current = audit(manifest, args.tex, args.workbook)
    write_report(manifest, games, metrics, current, args.output_dir)
    require(before == {p: sha256(p) for p in before}, "An input changed during the audit")
    print(f"PASS: {len(games)} games, {len(metrics)} aggregate rows and all deltas match the paper.")
    print(f"Pairs: {len(manifest['records']) // 2}; current workbook alone: {sum(r['Match'] for r in current)}/{len(current)} games.")
    print(f"Report: {args.output_dir / 'main_performance_seeds.xlsx'}")


if __name__ == "__main__":
    main()
