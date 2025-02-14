from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import data  
import re
import wandb

wandb.init(project="math_eval", name="s1.1-32B_AIME24_Q2")
model = LLM(
    "simplescaling/s1.1-32B",
    tensor_parallel_size=4,
)
tok = AutoTokenizer.from_pretrained("simplescaling/s1-32B")

stop_token_ids = tok("<|im_end|>")["input_ids"]

sampling_params = SamplingParams(
    max_tokens=32768,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
)

generator = data.data_generator('Maxwell-Jia/AIME_2024')  
correct = 0
for i in range(40):
    prompt, answer = generator.generate_answer_2()

    prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    o = model.generate(prompt, sampling_params=sampling_params)
    print(o[0].outputs[0].text)
    print(f'answer: {answer}')
    pattern = r"\\boxed\{(\d+)\}"
    matches = re.findall(pattern, o[0].outputs[0].text)
    print(f'matches: {matches} in iteration {i}')
    if matches[-1] == answer:
        correct += 1

print(f'correct: {correct} out of 40')
wandb.log({'correct': correct})
wandb.finish()