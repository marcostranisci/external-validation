"""Entry point: render the per-model, per-condition recall bar chart
(zero-shot / generated belief / verbalized belief).

Usage:
    python run_hs_recall_barplot.py
"""

from src.hs_recall_barplot import plot_recall_barplot


def main() -> None:
    out = plot_recall_barplot(
        generated_json="hs_detection/implicit_hate_all_models.json",
        verbalized_json="hs_detection/implicit_hate_verbalized_all_models.json",
        out_path="data_analysis/figures/hs_recall_barplot.png",
    )
    print(f"Saved {out} (and .pdf alongside it)")


if __name__ == "__main__":
    main()
