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