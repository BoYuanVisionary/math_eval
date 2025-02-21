from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re



from abc import ABC, abstractmethod

class BaseModel(ABC):
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = LLM(
            model_name,
            tensor_parallel_size=2,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.sampling_params = None

        
    @abstractmethod # preprocess the prompt
    def preprocess(self, prompt):
        pass
    
    def eval(self, dataset) -> str:
    
        correct = 0
        for i in range(len(dataset['train'])):
            prompt = dataset['train']['problem'][i]
            answer = dataset['train']['answer'][i]
            prompt = self.preprocess(prompt)
            o = self.model.generate(prompt, sampling_params=self.sampling_params)
            pattern = r"\\boxed\{(\d+)\}"
            matches = re.findall(pattern, o[0].outputs[0].text)
            print(f'answer: {answer}')
            print(f'matches: {matches}')
            if len(set(matches)) != 1:
                print(f'matches: {matches} not unique')
                print(f'current answer: {answer}')
            # remove the leading 0s
            if matches[-1].lstrip('0') == str(answer).lstrip('0'):
                correct += 1
            print(f'correct: {correct}, in {i+1} questions')

        print(f'correct: {correct}')
        return correct, len(dataset['train'])


class s1_model(BaseModel): # works for both s1 and s1.1
    
    def __init__(self, model_name):
        super().__init__(model_name)


        self.sampling_params = SamplingParams(
            max_tokens=32768,
            min_tokens=0,
            stop_token_ids=self.tokenizer("<|im_end|>")["input_ids"],
        )
        
    def preprocess(self, prompt) -> str:
        prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        return prompt
        
       
    
class LIMO_model(BaseModel):

    def __init__(self, model_name):
        super().__init__(model_name)
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=32768,
            top_p=0.95,
        )

    def preprocess(self, prompt) -> str:
        
        messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    



