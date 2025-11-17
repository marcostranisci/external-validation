from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils import estimate_uncertainty
from lm_polygraph.estimators import *






MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"

class MyModel:
    def __init__(self,model_path,estimator=MeanTokenEntropy()):
        base_model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.estimator = estimator
        generation_params = GenerationParameters()
        generation_params.temperature = 0.1
        generation_params.do_sample = False
        generation_params.max_new_tokens = 20

        self.model = WhiteboxModel(base_model, self.tokenizer, model_path=model_path,generation_parameters=generation_params)
    
    def estimate_uncertainty(self,input_text):
        
        #prompt = self.tokenizer.apply_chat_template(input_text, tokenize=False)
        ue = estimate_uncertainty(self.model, self.estimator, input_text=input_text)
        return ue


message = {"role": "user",
          "task": "you are a participant to a psychological survey",
        "instruction": "Answer the following question with one value from the following options",
          "options": "1,2,3,4,5. 1 == strongly disagree, 5 == strongly agree",
          "output_format": "return a single value from the options",
          "content": "Is it right to treat some people differently than others?",
          "output": "output"
}


prompt = f"""
Task: {message['task']}

Instruction: {message['instruction']}

Options: {message['options']}

Question: {message['content']}

Output format: {message['output_template']}

Answer only in JSON. No extra text.
"""

mod = MyModel(MODEL_PATH,MeanTokenEntropy())
ue = estimate_uncertainty(mod, input_text=prompt)
