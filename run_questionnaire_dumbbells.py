"""Entry point: render MFT/PVQ questionnaire-vs-interview dumbbell charts,
side by side.

Usage:
    python run_questionnaire_dumbbells.py
"""

from src.questionnaire_dumbbells import plot_score_dumbbells


def main() -> None:
    out = plot_score_dumbbells(
        mft_dir="data_analysis/processed/mft",
        pvq_dir="data_analysis/processed/pvq",
        out_path="data_analysis/figures/questionnaire_vs_interview_dumbbells.png",
    )
    print(f"Saved {out} (and .pdf alongside it)")


if __name__ == "__main__":
    main()
