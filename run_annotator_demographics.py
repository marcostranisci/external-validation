"""Entry point: check whether annotators sharing a demographic (gender,
continent of birth) agree with each other more than annotators who don't,
when evaluating a model's replies — computed separately per model (i.e. per
QUESTNNR group of 9 annotators in the merged exports).

Usage:
    python run_annotator_demographics.py
"""

import logging

import pandas as pd

from src.questionnaire_analysis import AnnotatorSurveyAnalyzer

# Every country of birth present in mf_merged.csv / pv_merged.csv, bucketed
# into the three continents the annotator pool was stratified on.
CONTINENT = {
    "United Kingdom": "Europe", "Ireland": "Europe",
    "South Africa": "Africa", "Nigeria": "Africa", "Kenya": "Africa", "Zimbabwe": "Africa",
    "India": "Asia", "Pakistan": "Asia", "Philippines": "Asia", "Hong Kong": "Asia",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    analyzer = AnnotatorSurveyAnalyzer(output_dir="data_analysis")

    for prefix, filepath, survey_glob in [
        ("MF", "surveys/mf_merged.csv", "surveys/mft/*.csv"),
        ("PV", "surveys/pv_merged.csv", "surveys/pvq/*.csv"),
    ]:
        df = pd.read_csv(filepath)
        df["continent"] = df["Country of birth"].map(CONTINENT)
        unmapped = df["continent"].isna().sum()
        if unmapped:
            logging.warning(f"[{filepath}] {unmapped} row(s) with an unmapped country of birth")

        mapping = analyzer.infer_model_mapping(df, prefix, survey_glob)
        analyzer.demographic_agreement(df, prefix, ["Gender", "continent"], model_mapping=mapping)

    out_folder = analyzer.output_dir / "annotators"
    out_folder.mkdir(parents=True, exist_ok=True)
    with open(out_folder / "demographic_agreement_warnings.log", "w") as fh:
        fh.write("\n".join(analyzer.warnings) + "\n" if analyzer.warnings else "No warnings.\n")

    print(f"Done. Results written to {analyzer.output_dir}/annotators/")
    if analyzer.warnings:
        print(f"{len(analyzer.warnings)} warning(s) - see "
              f"data_analysis/annotators/demographic_agreement_warnings.log")


if __name__ == "__main__":
    main()
