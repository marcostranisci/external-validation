"""Entry point: correlate annotators' own questionnaire replies against how
they evaluate models' replies, for the MF and PV merged annotator exports.

Usage:
    python run_annotator_analysis.py
"""

import logging

from src.questionnaire_analysis import AnnotatorSurveyAnalyzer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    analyzer = AnnotatorSurveyAnalyzer(output_dir="data_analysis")
    analyzer.run_all({
        "MF": "surveys/mf_merged.csv",
        "PV": "surveys/pv_merged.csv",
    })
    print(f"Done. Results written to {analyzer.output_dir}/annotators/")
    if analyzer.warnings:
        print(f"{len(analyzer.warnings)} data quality warning(s) - "
              f"see data_analysis/annotators/data_quality_warnings.log")


if __name__ == "__main__":
    main()
