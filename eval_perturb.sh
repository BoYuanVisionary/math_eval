# os.environ["OPENAI_API_KEY"] = "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"
export OPENAI_API_KEY="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"

python eval_perturb.py --model "simplescaling/s1-32B" --run_times 3 --task "AIME24" --perturbation "add_word"
python eval_perturb.py --model "simplescaling/s1-32B" --run_times 3 --task "AIME24" --perturbation "remove_word"

python eval_perturb.py --model "simplescaling/s1-32B" --run_times 3 --task "AIME25" --perturbation "add_word"
python eval_perturb.py --model "simplescaling/s1-32B" --run_times 3 --task "AIME25" --perturbation "remove_word"

python eval_perturb.py --model "simplescaling/s1.1-32B" --run_times 3 --task "AIME24" --perturbation "add_word"
python eval_perturb.py --model "simplescaling/s1.1-32B" --run_times 3 --task "AIME24" --perturbation "remove_word"

python eval_perturb.py --model "simplescaling/s1.1-32B" --run_times 3 --task "AIME25" --perturbation "add_word"
python eval_perturb.py --model "simplescaling/s1.1-32B" --run_times 3 --task "AIME25" --perturbation "remove_word"


python eval_perturb.py --model "GAIR/LIMO" --run_times 3 --task "AIME24" --perturbation "add_word"
python eval_perturb.py --model "GAIR/LIMO" --run_times 3 --task "AIME24" --perturbation "remove_word"

python eval_perturb.py --model "GAIR/LIMO" --run_times 3 --task "AIME25" --perturbation "add_word"
python eval_perturb.py --model "GAIR/LIMO" --run_times 3 --task "AIME25" --perturbation "remove_word"