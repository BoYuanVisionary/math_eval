from data import data_generator
import json
from model import *
import wandb
import argparse

# USe the first two GPUS for inference
# need at least 2 GPUs for 32B model
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="simplescaling/s1-32B")
    parser.add_argument("--run_times", type=int, default=3)
    parser.add_argument("--task", type=str, default='AIME24')
    parser.add_argument("--perturbation", type=str, default='add_word')
    return parser.parse_args()

args = get_args()
description = f'{args.model} on {args.task} with {args.run_times} runs'
seeds = range(args.run_times) 
run = wandb.init(project="math_eval",name=description)  
artifact = wandb.Artifact("math-problems-perturbation", type="dataset")


if args.model in ['simplescaling/s1-32B','simplescaling/s1.1-32B']:
    llm = s1_model(args.model)
elif args.model == 'GAIR/LIMO':
    llm = LIMO_model(args.model)
else:
    raise ValueError(f'Model {args.model} not supported')

if args.task == 'AIME24':
    dataset_name = 'AI-MO/aimo-validation-aime'
elif args.task == 'AIME25':
    dataset_name = 'BBexist/AIME25'
else:
    raise ValueError(f'Task {args.task} not supported')



for seed in seeds:
    print(f'experiment on {description}')
    dataGenerator = data_generator(dataset_name,seed) # this returns the same dataset when seed is the same
    
    if args.perturbation == 'add_word':
        perturbed_dataset = dataGenerator.perturbation_add_word()
    elif args.perturbation == 'remove_word':
        perturbed_dataset = dataGenerator.perturbation_remove_word()
    else:
        raise ValueError(f'Perturbation {args.perturbation} not supported')
    

    problems = perturbed_dataset['train']['problem']
    target_words = dataGenerator.target_words
    dataGenerator.clean_target_words() # this should be called right after perturbation and  target_words
    items = [{'problem': problems[i], 'target_word': target_words[i]} for i in range(len(problems))]
    with open(f'./logs/{args.perturbation}_dataset_{seed}.json', 'w') as f:
        json.dump(items, f)
    artifact.add_file(f'./logs/{args.perturbation}_dataset_{seed}.json')
    run.log_artifact(artifact)
    correct, total = llm.eval(perturbed_dataset)
    print(f'{args.perturbation} correct: {correct}, total: {total}')
    wandb.log({f'{args.perturbation}': correct/total})
    
print(f'experiment on {description} finished')
print('--------------------------------')
wandb.finish()