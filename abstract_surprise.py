import os
import unittest
from pathlib import Path

import numpy as np
import scipy
#from sentence_transformers import SentenceTransformer


def calculate_surprise():

    data_dir = Path(__file__).parent
    abstract_t0=os.path.join(data_dir,"data","Abstract_t0.txt") #past abstract
    abstract_t1=os.path.join(data_dir,"data","Abstract_t1.txt") #present abstract
    abstract_t2 = os.path.join(data_dir, "data", "Abstract_t2.txt")  #future abstract


    with open(abstract_t0, 'r') as file:
        content_0 = file.read()

    with open(abstract_t1, 'r') as file:
        content_1 = file.read()

    with open(abstract_t2, 'r') as file:
        content_2 = file.read()

    # sanity check, this should be zero
    kl_divergence_should_be_zero=calculate_surprise_between_two_strings(content_1, content_1)

    #how surprising is content_1 given content_0?
    Novelty_of_abs_1_given_abs_0 = calculate_surprise_between_two_strings(content_0, content_1)

    #how surprising is content_2 given content_1?
    Novelty_of_abs_2_given_abs_1 = calculate_surprise_between_two_strings(content_1, content_2)
    Novelty_of_abs_2_given_abs_0 = calculate_surprise_between_two_strings(content_0, content_2)

    #from "Individuals, institutions, and innovation in the debates
    #        of the French Revolution" paper


    # Novelty: surprise of now, given the past  (how much now is not like the past)
    # Transience: surprise of now, given the future (how much now is not like the future)
    Transience_of_abs_1 = calculate_surprise_between_two_strings(content_2, content_1)

    # Resonance is novelty minus transience
    Resonance_of_abs_1 = Novelty_of_abs_1_given_abs_0-Transience_of_abs_1
    print("Resonance_of_abs_1:",Resonance_of_abs_1)

    #Impact. Is Novelty(2|1) < Novelty(2|0) ?
    #If  abs 0 predicts the future just as well as abs 1, then abs 1 didnt have much impact!
    Impact_of_abs_1 = Novelty_of_abs_2_given_abs_0 - Novelty_of_abs_2_given_abs_1
    print("Impact_of_abs_1:",Impact_of_abs_1)

def calculate_surprise_between_two_strings(training_line2, testing_line1, ):

    words1 = testing_line1.split()
    words2 = training_line2.split()
    all_words = list(set(words1 + words2))

    epsilon = 1e-10
    p = np.array([words1.count(w) for w in all_words], dtype=float)
    p += epsilon
    p /= np.sum(p)  # Normalize to sum to 1

    q = np.array([words2.count(w) for w in all_words], dtype=float)
    q += epsilon
    q /= np.sum(q)  # Normalize to sum to 1

    # 4. Calculate KL divergence using scipy
    # KL(p || q): Surprise of expecting line2 but seeing line1
    kl_divergence = scipy.stats.entropy(pk=p, qk=q)
    print(f"KL Divergence (surprise): {kl_divergence:.4f}")
    return kl_divergence



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    calculate_surprise()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
