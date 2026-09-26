"""Entry point: compare belief-steering by free-text opinion
(data_analysis/hs_detection/) against steering by the verbalized
questionnaire item/Likert score itself
(data_analysis/hs_detection_verbalized_shared_baseline/) — using the
shared-baseline run, so both conditions' recall/flip/significance numbers
are computed against the same (free-text file's) zero_shot predictions —
there is no own-baseline verbalized comparison in this codebase, since
that would confound belief content with a prompt-wording difference
between the two files.

Run run_hs_detection_analysis.py and
run_hs_detection_verbalized_shared_baseline_analysis.py first.

Usage:
    python run_hs_detection_comparison_shared_baseline.py
"""

from src.hs_detection_comparison import SteeringConditionComparison


def main() -> None:
    cmp = SteeringConditionComparison(
        dir_a="data_analysis/hs_detection", label_a="free_text",
        dir_b="data_analysis/hs_detection_verbalized_shared_baseline", label_b="verbalized",
        output_dir="data_analysis/hs_detection_comparison_shared_baseline",
    )
    cmp.run_all()
    print(f"Done. Results written to {cmp.output_dir}/")


if __name__ == "__main__":
    main()
