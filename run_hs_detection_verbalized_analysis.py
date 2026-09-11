"""Entry point: same analysis as run_hs_detection_analysis.py, but on the
verbalized-belief steering data (models steered by their own verbalized
Likert score on each item, e.g. '"X" describes you extremely well.',
rather than by their free-text opinion).

Usage:
    python run_hs_detection_verbalized_analysis.py
"""

from src.hs_detection_analysis import HateSpeechSteeringAnalyzer


def main() -> None:
    analyzer = HateSpeechSteeringAnalyzer(
        json_path="hs_detection/implicit_hate_verbalized_all_models.json",
        output_dir="data_analysis",
        output_subdir="hs_detection_verbalized",
    )
    analyzer.run_all(pvq_mapping_path="hs_detection/pvq_items_mapping.csv")
    print(f"Done. Results written to {analyzer.output_dir}/{analyzer.output_subdir}/")


if __name__ == "__main__":
    main()
