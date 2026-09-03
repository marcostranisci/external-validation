"""Entry point: does questionnaire-belief steering shift hate-speech
detection recall, and does the shift differ between models?

Usage:
    python run_hs_detection_analysis.py
"""

from src.hs_detection_analysis import HateSpeechSteeringAnalyzer


def main() -> None:
    analyzer = HateSpeechSteeringAnalyzer(json_path="hs_detection/implicit_hate_all_models.json")
    analyzer.run_all()
    print(f"Done. Results written to {analyzer.output_dir}/hs_detection/")


if __name__ == "__main__":
    main()
