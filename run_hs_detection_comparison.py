"""Entry point: compare belief-steering by free-text opinion
(data_analysis/hs_detection/) against steering by the verbalized
questionnaire item/Likert score itself (data_analysis/hs_detection_verbalized/).

Run run_hs_detection_analysis.py and run_hs_detection_verbalized_analysis.py
first.

Usage:
    python run_hs_detection_comparison.py
"""

from src.hs_detection_comparison import SteeringConditionComparison


def main() -> None:
    cmp = SteeringConditionComparison(
        dir_a="data_analysis/hs_detection", label_a="free_text",
        dir_b="data_analysis/hs_detection_verbalized", label_b="verbalized",
        output_dir="data_analysis/hs_detection_comparison",
    )
    cmp.run_all()
    print(f"Done. Results written to {cmp.output_dir}/")


if __name__ == "__main__":
    main()
