import argparse

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, Side


def format_score(value):
    if pd.isna(value) or str(value).strip() in {"", "N/A", "nan"}:
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def main():
    parser = argparse.ArgumentParser(description="Convert Drama WandB CSV to an O/X comparison workbook.")
    parser.add_argument("--input", default="drama_wandb_runs.csv")
    parser.add_argument("--output", default="drama_results.xlsx")
    args = parser.parse_args()

    runs = pd.read_csv(args.input)
    if runs.empty:
        raise ValueError("The input CSV contains no O/X runs.")

    runs["Created At"] = pd.to_datetime(runs["Created At"], utc=True, errors="coerce").dt.tz_localize(None)
    runs["Eval Return Numeric"] = pd.to_numeric(runs["Eval Return"], errors="coerce")
    runs = runs[runs["Retrieval"].isin(["O", "X"])].copy()
    if runs.empty:
        raise ValueError("The input CSV contains no O/X runs.")

    # Keep the newest run when the same game, seed, and mode was logged more than once.
    runs = runs.sort_values("Created At").drop_duplicates(
        ["Game", "Seed", "Retrieval"], keep="last"
    )
    results = runs.pivot_table(
        index=["Game", "Retrieval"],
        columns="Seed",
        values="Eval Return Numeric",
        aggfunc="first",
    ).reindex(pd.MultiIndex.from_product(
        [sorted(runs["Game"].unique()), ["X", "O"]],
        names=["Game", "Retrieval"],
    ))
    results.columns.name = None
    results = results.reset_index()
    seed_columns = [column for column in results.columns if column not in ["Game", "Retrieval"]]
    results = results[["Game", "Retrieval"] + sorted(seed_columns)]

    detail = runs.drop(columns=["Eval Return Numeric"])
    detail["Eval Return"] = detail["Eval Return"].map(format_score)
    detail["Eval Normalized Return"] = detail["Eval Normalized Return"].map(format_score)
    for column in results.columns[2:]:
        results[column] = results[column].map(format_score)

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Results", index=False)
        detail.to_excel(writer, sheet_name="Runs", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.font = Font(name="Calibri", size=12, bold=cell.row == 1)
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
            for column_cells in worksheet.columns:
                width = max(len(str(cell.value or "")) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(width + 2, 12), 36)

        worksheet = writer.book["Results"]
        top_side = Side(style="medium", color="000000")
        previous_game = None
        for row_index in range(2, worksheet.max_row + 1):
            game = worksheet.cell(row=row_index, column=1).value
            if game != previous_game:
                for column_index in range(1, worksheet.max_column + 1):
                    cell = worksheet.cell(row=row_index, column=column_index)
                    cell.border = Border(
                        top=top_side,
                        left=cell.border.left,
                        right=cell.border.right,
                        bottom=cell.border.bottom,
                    )
                previous_game = game

    print(f"Successfully saved to {args.output}")


if __name__ == "__main__":
    main()
