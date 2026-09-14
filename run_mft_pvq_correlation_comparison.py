"""Entry point: is between-model agreement on opinion/external_opinion
systematically stronger on PVQ than on MFT? Compares the two
questionnaires' already-computed between_model_correlations_*.csv files
via a Fisher-z Mann-Whitney U / Welch t-test.

Run run_questionnaire_analysis.py first.

Usage:
    python run_mft_pvq_correlation_comparison.py
"""

from src.questionnaire_analysis import QuestionnaireAnalyzer


def main() -> None:
    analyzer = QuestionnaireAnalyzer(output_dir="data_analysis")
    result = analyzer.compare_between_model_correlations("mft", "pvq")
    print(result.to_string(index=False))
    print(f"\nDone. Results written to {analyzer.output_dir}/between_model_correlations_mft_vs_pvq.csv")


if __name__ == "__main__":
    main()
