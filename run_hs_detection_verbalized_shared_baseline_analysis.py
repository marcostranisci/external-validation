"""Entry point: HateSpeechSteeringAnalyzer on the verbalized-belief
steering JSON (hs_detection/implicit_hate_verbalized_all_models.json),
with every model's zero_shot baseline replaced by that model's zero_shot
predictions from the free-text file (hs_detection/implicit_hate_all_models.json)
instead of the verbalized file's own zero_shot run.

This controls for a wording difference in zero_shot_prompt/belief_prompt
between the two files (documented in data_analysis/README.md, "does
steering method matter?") by forcing both steering conditions to share
one baseline. There is no own-baseline verbalized analysis in this
codebase to compare against — comparing the verbalized condition against
its own zero-shot would confound belief content with prompt wording, so
every downstream recall/flip/significance number for the verbalized
condition is computed this way.

Usage:
    python run_hs_detection_verbalized_shared_baseline_analysis.py
"""

from src.hs_detection_analysis import HateSpeechSteeringAnalyzer


def main() -> None:
    analyzer = HateSpeechSteeringAnalyzer(
        json_path="hs_detection/implicit_hate_verbalized_all_models.json",
        baseline_json_path="hs_detection/implicit_hate_all_models.json",
        output_dir="data_analysis",
        output_subdir="hs_detection_verbalized_shared_baseline",
    )
    analyzer.run_all(pvq_mapping_path="hs_detection/pvq_items_mapping.csv")
    print(f"Done. Results written to {analyzer.output_dir}/{analyzer.output_subdir}/")


if __name__ == "__main__":
    main()
