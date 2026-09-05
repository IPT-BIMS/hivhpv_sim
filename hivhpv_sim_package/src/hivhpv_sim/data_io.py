"""I/O helpers: multi-country project folder setup, Excel I/O."""
from pathlib import Path
from typing import Union
import re
import pandas as pd

TABLES_PATH = None
FIGURES_PATH = None
CALIBRATION_FIGURES_PATH = None
STRATEGY_RESULTS_PATH = None


def make_project_dirs(country_name: str, project_root=None):
    """Create multi-country folder tree and bind module globals."""
    global TABLES_PATH, FIGURES_PATH, CALIBRATION_FIGURES_PATH, STRATEGY_RESULTS_PATH
    if project_root is None:
        project_root = Path.cwd()
    project_root = Path(project_root)
    cal_results  = project_root / "03_calibration_results" / country_name
    cal_tables   = cal_results / "tables"
    cal_figures  = project_root / "04_figures" / country_name / "calibration"
    strat_results = project_root / "05_strategy_results" / country_name
    for p in [cal_results, cal_tables, cal_figures, strat_results]:
        p.mkdir(parents=True, exist_ok=True)
    TABLES_PATH = cal_tables
    FIGURES_PATH = cal_figures
    CALIBRATION_FIGURES_PATH = cal_figures
    STRATEGY_RESULTS_PATH = strat_results
    return {"CALIBRATION_RESULTS_PATH": cal_results, "CALIBRATION_TABLES_PATH": cal_tables,
             "CALIBRATION_FIGURES_PATH": cal_figures, "STRATEGY_RESULTS_PATH": strat_results}


def make_output_dirs(root_name: str = "test"):
    global TABLES_PATH, FIGURES_PATH
    base = Path(root_name) / "results"
    TABLES_PATH  = base / "tables"
    FIGURES_PATH = base / "figures"
    TABLES_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    return base, TABLES_PATH, FIGURES_PATH


def load_life_tables(country_name: str, country_input_dir=None):
    if country_input_dir is None:
        path = Path(f"{country_name}_life_table.xlsx")
    else:
        path = Path(country_input_dir) / f"{country_name}_life_table.xlsx"
    df_field = pd.read_excel(path, sheet_name="cc")
    if "age" not in df_field.columns and "age_group" in df_field.columns:
        df_field = df_field.rename(columns={"age_group": "age"})
    return {"demography": pd.read_excel(path, sheet_name="demography"),
             "qx_hiv":    pd.read_excel(path, sheet_name="qx_HIV"),
             "field_data": df_field}

def save_table_excel(
    df: pd.DataFrame,
    writer_or_filename: Union[pd.ExcelWriter, str, Path],
    sheet_name: str = "Sheet1",
    *,
    index: bool = False,
) -> None:
    """Save a DataFrame to an Excel file or an existing ExcelWriter."""

    if isinstance(writer_or_filename, pd.ExcelWriter):
        df.to_excel(
            writer_or_filename,
            sheet_name=sheet_name,
            index=index,
        )
        return

    filename = Path(writer_or_filename).stem
    filepath = TABLES_PATH / f"{filename}.xlsx"

    TABLES_PATH.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        with pd.ExcelWriter(
            filepath,
            mode="a",
            engine="openpyxl",
            if_sheet_exists="replace",
        ) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)
    else:
        with pd.ExcelWriter(
            filepath,
            mode="w",
            engine="openpyxl",
        ) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)

    print(f"Table saved to {filepath} (sheet: {sheet_name})")

def save_figure(fig, filename: str, dpi: int = 300) -> None:
    """Save a Matplotlib figure in the figures directory."""

    filepath = FIGURES_PATH / f"{Path(filename).stem}.png"
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved to {filepath}")

def _excel_safe_sheet_name(name: str, used: set[str]) -> str:
    """
    Excel constraints:
      - <= 31 chars
      - cannot contain: []:*?/\\
      - must be unique
    """
    s = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(name)).strip()
    if not s:
        s = "Sheet"
    s = s[:31]

    if s not in used:
        used.add(s)
        return s

    base = s[:28]  # room for _01, _02, ...
    k = 1
    while True:
        cand = f"{base}_{k:02d}"[:31]
        if cand not in used:
            used.add(cand)
            return cand
        k += 1

