"""Entry point: run the questionnaire analysis pipeline on the mft and pvq surveys.

Usage:
    python run_questionnaire_analysis.py
"""

import logging

from src.questionnaire_analysis import QuestionnaireAnalyzer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    analyzer = QuestionnaireAnalyzer(surveys_dir="surveys", output_dir="data_analysis")
    analyzer.run_all(["mft", "pvq"])
    print(f"Done. Results written to {analyzer.output_dir}/")
    if analyzer.warnings:
        print(f"{len(analyzer.warnings)} data quality warning(s) - see data_analysis/data_quality_warnings.log")


if __name__ == "__main__":
    main()
