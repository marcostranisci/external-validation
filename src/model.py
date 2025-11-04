'''from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils import estimate_uncertainty






MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"

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
        return ue'''

'''message = {
    "role": "user",
    "task": "you are a participant to a psychological survey",
    "instruction": "Answer the following question with one value from the following options",
    "options": "1,2,3,4,5. 1 == strongly disagree, 5 == strongly agree",
    "output_template": "the answer should follow this template {{\"answer\": option}}",
    "content": "Is it right to treat some people differently than others?",
    "output": "output"
}

# ====== PROMPT TEMPLATE ======
prompt = f"""
Task: {message['task']}

Instruction: {message['instruction']}

Options: {message['options']}

Question: {message['content']}

Output format: {message['output_template']}

Answer only in JSON. No extra text.
"""

mod = Model(MODEL_PATH,MeanTokenEntropy())

print(mod.estimate_uncertainty(prompt))
'''
'''base_model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = tokenizer.apply_chat_template(message, tokenize=False)



ue_method = MeanTokenEntropy()

max_new_tokens = 100
generation_params = GenerationParameters()
generation_params.temperature = 0.2
generation_params.do_sample = True
generation_params.max_new_tokens = max_new_tokens

model = WhiteboxModel(base_model, tokenizer, model_path=model_path,generation_parameters=generation_params)

model.generation_parameters.max_new_tokens = max_new_tokens

input_text = "Answer the following question with one value from the following options: 1,2,3,4,5. 1 == strongly disagree; 5 == strongly agree. Is it right to treat some people differently than others?"
ue = estimate_uncertainty(model, ue_method, input_text=prompt)
print(ue)

class Model:
    def __init__(self,model_path,estimator):
        base_model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.estimator = estimator
        generation_params = GenerationParameters()
        generation_params.temperature = 0.2
        generation_params.do_sample = True
        generation_params.max_new_tokens = 100

        self.model = WhiteboxModel(base_model, self.tokenizer, model_path=model_path,generation_parameters=generation_params)
    
    def estimate_uncertainty(self,ue_method,input_text):
        
        prompt = self.tokenizer.apply_chat_template(input_text, tokenize=False)
        ue = estimate_uncertainty(self.model, self.estimator, input_text=prompt)
        return ue

message = [{"role": "user",
          "task": "you are a participant to a psychological survey",
        "instruction": "Answer the following question with one value from the following options",
          "options": "1,2,3,4,5. 1 == strongly disagree, 5 == strongly agree",
          "output_format": "return a single value from the options",
          "content": "Is it right to treat some people differently than others?",
          "output": "output"
}
]


mod = Model(MODEL_PATH,MeanTokenEntropy())'''