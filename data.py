from datasets import load_dataset, Dataset
import random
import sympy as sp
from fractions import Fraction
from openai import OpenAI, APIError, RateLimitError, APIConnectionError


class data_generator:
    
    def __init__(self, dataset_name: str, seed: int=42):
        self.dataset = load_dataset(dataset_name)
        if dataset_name == 'AI-MO/aimo-validation-aime':
            # filter out the examples that are not from 2024
            self.dataset = self.dataset.filter(lambda example: '2024' in example['url'])
        self._rand = random.Random(seed)
        self.target_words = []
        self.forbidden_symbols =  {'$', '+', '-', '=', '\\', '(', ')', '{', '}', '[', ']', '^', '_'}
     
      
    def clean_target_words(self):
        self.target_words = []
    
    def _remove_random_word_in_string(self, s: str) -> str:
        words = s.split()
        if not words: 
            return s
        while True:
            idx = self._rand.randrange(len(words))
            target_word = words[idx]
            # check if the target word is a valid word
            if (not any(letter.isdigit() for letter in target_word)) \
                and (not any(letter in self.forbidden_symbols for letter in target_word)):
                break
        words.pop(idx)
        self.target_words.append(target_word)
        print(f"removed word: {target_word}")
        return " ".join(words)

    def _add_random_word_in_string(self, s: str) -> str:
        words = s.split()
        if not words: 
            return s
        while True:
            idx = self._rand.randrange(len(words))
            target_word = words[idx]
            # check if the target word is a valid word
            if (not any(letter.isdigit() for letter in target_word)) \
                and (not any(letter in self.forbidden_symbols for letter in target_word)):
                break
        words.insert(idx, words[idx])
        self.target_words.append(target_word)
        return " ".join(words)
    
    def _rephrase_problem(self, problem: str, seed: int) -> str:
        # use openai api to rephrase the problem
        import os
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type
        )
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(
                (RateLimitError, APIConnectionError, APIError)
            )
        )
        def create_chat_completion_with_retry():
            return client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": f"Rephrase this math problem while keeping all numerical values and mathematical expressions unchanged. Output only the rephrased problem:\n\n{problem}"
                }],
                max_tokens=1000,
                seed=seed
            )

        try:
            response = create_chat_completion_with_retry()
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Failed to rephrase problem after retries: {str(e)}")
            return problem 
        
        # To do: have another LLM check if the rephrased problem is correct 
        
    def perturbation_remove_word(self) -> Dataset:
        dataset_remove_one = self.dataset.map(lambda example: {"problem": self._remove_random_word_in_string(example["problem"])}, load_from_cache_file=False)
        return dataset_remove_one
    
    def perturbation_add_word(self) -> Dataset:
        dataset_add_one = self.dataset.map(lambda example: {"problem": self._add_random_word_in_string(example["problem"])}, load_from_cache_file=False)
        return dataset_add_one
    
    def repharse_dataset(self) -> Dataset:
        dataset_rephrase = self.dataset.map(lambda example: {"problem": self._rephrase_problem(example["problem"])}, load_from_cache_file=False)
        return dataset_rephrase
    
    # The folllowing functions should follow the specific question format
    
    def generate_answer_2(self,x_sym=0):  # data generation function for AIME 24 I Problem 12

        # only consider rational x_sym if initial value is given
        if x_sym == 0:
            while x_sym == 0 or x_sym==1:
                denominator = self._rand.randint(2, 10)
                numerator = self._rand.randint(0, denominator)
                x_frac = Fraction(numerator, denominator)
                x_sym  = sp.Rational(x_frac.numerator, x_frac.denominator)
                
        x_frac = Fraction(x_sym)        
        x_latex = sp.latex(x_sym) 
        y_expr = sp.sqrt(1 - x_sym**2)
        y_expr = sp.nsimplify(y_expr) 
        y_latex = sp.latex(y_expr)
        
            
        OC_square = x_frac ** 6 + y_expr ** 6
        frac_oc_square = Fraction(OC_square)
        p,q = frac_oc_square.numerator, frac_oc_square.denominator
        real_answer = p + q 
        
        templete = """Let $O(0,0), A({x_latex}, 0),$ and $B(0, {y_latex})$ be points in the coordinate plane. 
        Let $\\mathcal{{F}}$ be the family of segments $\\overline\{{PQ\}}$ of unit length lying in the first quadrant with $P$ on the $x$-axis and $Q$ on the $y$-axis. There is a unique point $C$ on $\\overline{{AB}}$, distinct from $A$ and $B$, that does not belong to any segment from $\\mathcal{{F}}$ other than $\\overline{{AB}}$. 
        Then $OC^2 = \\tfrac{{p}}{{q}}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$.
        """
        prompt = templete.format(x_latex=x_latex, y_latex=y_latex)
        print(f"{x_latex}, {y_latex}")

        return prompt, real_answer
 
