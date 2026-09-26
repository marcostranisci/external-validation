"""Entry point: render pairwise inter-model agreement heatmaps for the three
hate-speech classification conditions (zero-shot, generated belief,
verbalized belief).

Usage:
    python run_hs_agreement_heatmaps.py
"""

from src.hs_agreement_heatmaps import plot_agreement_heatmaps, plot_agreement_heatmaps_by_questionnaire


def main() -> None:
    out = plot_agreement_heatmaps(
        zero_shot_csv="data_analysis/hs_detection/pairwise_model_agreement_zero_shot.csv",
        generated_csv="data_analysis/hs_detection/pairwise_model_agreement_steered.csv",
        verbalized_csv="data_analysis/hs_detection_verbalized_shared_baseline/pairwise_model_agreement_steered.csv",
        out_path="data_analysis/figures/hs_agreement_heatmaps.png",
    )
    print(f"Saved {out} (and .pdf alongside it)")

    out2 = plot_agreement_heatmaps_by_questionnaire(
        generated_json="hs_detection/implicit_hate_all_models.json",
        verbalized_json="hs_detection/implicit_hate_verbalized_all_models.json",
        out_path="data_analysis/figures/hs_agreement_heatmaps_by_questionnaire.png",
    )
    print(f"Saved {out2} (and .pdf alongside it)")


if __name__ == "__main__":
    main()
