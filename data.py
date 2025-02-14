from datasets import load_dataset
import random
import sympy as sp
from fractions import Fraction
random.seed(42)


class data_generator:
    def __init__(self, dataset_name: str):
        self.dataset = load_dataset(dataset_name)
        
        
    def remove_random_word_in_string(self, s: str) -> str:

        words = s.split()
        if not words: 
            return s

        idx = random.randrange(len(words))
        target_word = words[idx]    
        words.pop(idx)
        # print(f'the word {target_word} is removed')
        return " ".join(words)

    def add_random_word_in_string(self, s: str) -> str:

        words = s.split()
        if not words: 
            return s

        idx = random.randrange(len(words))
        target_word = words[idx]
        words.insert(idx, words[idx])
        # print(f'the word {target_word} is repeated')
        return " ".join(words)    

    def generate_answer_2(self,x_sym=0):  # data generation function for question 2

        # only consider rational x_sym if initial value is given
        if x_sym == 0:
            while x_sym == 0 or x_sym==1:
                denominator = random.randint(2, 10)
                numerator = random.randint(0, denominator)
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
 
