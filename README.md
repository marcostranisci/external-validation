# Internal Validation First, External Validation follows

This is the repository of the research xyz

### How To

1. create a conda environment with python 3.10
2. install requirements.txt

### the experiments
This first draft of the experiment involves 3 experiments that you can find in 'src/tasks.py'

* **mft_questionnaire:** it asks the LLM to fill the MFQ-2 questionnaire to determine its moral stance
* **mft_social_media:** it asks the LLM to identify the presence of moral foundations in social media messages
* **mft_offensiveness:** it asks the LLM to rank the offensiveness of a set of posts annotated by an annotator with a given moral stance.

### configuration
give a look at 'config.yml' to check the configuration of each experiments (they are very shallow and straightforward)