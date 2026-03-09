from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
OUT_DIR = BASE_DIR / "docs" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_table_png(csv_path: Path, out_path: Path, title: str):
    if not csv_path.exists():
        print(f"skip (missing): {csv_path}")
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.45 * (len(df) + 2))))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"saved: {out_path}")


def main():
    save_table_png(REPORT_DIR / "model_comparison.csv", OUT_DIR / "model_comparison_table.png", "Model Comparison")
    save_table_png(REPORT_DIR / "cv_metrics.csv", OUT_DIR / "cv_metrics_table.png", "TimeSeries CV Metrics")


if __name__ == "__main__":
    main()
