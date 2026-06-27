import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import json
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from sentence_transformers import SentenceTransformer


def calculate_surprise():

    git_data_dir = Path(__file__).parent
    big_data_dir = "/Users/tdunn/Data/SoS"
    #paper_to_test = 'W2141394518' #start with 'Lorenz 1963'
    paper_to_test = 'W119052030' #start with a random paper

    ThreeBreakthroughPaper_References_Citations_Path = (
        os.path.join(git_data_dir, "data", "ThreeBreakthroughPaper_References_Citations.txt"))  # future abstract
    RandomPaper_References_Citations_Path = (
        os.path.join(git_data_dir, "data", "YEARRandomPaper_References_Citations.txt"))  # future abstract

    Paperid_Title_Abstract_Path = (
        os.path.join(big_data_dir, "Paperid_Title_Abstract.txt"))  # future abstract

    InformationTheoryCases = {
        'W2141394518': 'Lorenz 1963',
        'W2126466006': 'DNA 1953',
        'W2126160338': 'Turing 1936'
    }

    ThreeBreakthroughs_R = {}
    ThreeBreakthroughs_C = {}
    with open(ThreeBreakthroughPaper_References_Citations_Path, 'r') as f:
        for line in f:
            line = json.loads(line.strip('\n'))
            ThreeBreakthroughs_R[line['id']] = line['references']
            ThreeBreakthroughs_C[line['id']] = line['citations']

    RandomPaper_R = {}
    RandomPaper_C = {}
    RandomPaper_Y = {}
    for y in [1936, 1953, 1963]:
        with open(RandomPaper_References_Citations_Path.replace("YEAR",str(y)), 'r') as f:
            for line in f:
                line = json.loads(line.strip('\n'))
                RandomPaper_R[line['id']] = line['references']
                RandomPaper_C[line['id']] = line['citations']
                RandomPaper_Y[line['id']] = y

    PaperTitle = {}
    PaperAbstract = {}
    with open(Paperid_Title_Abstract_Path, 'r') as f:
        for line in f:
            line = json.loads(line.strip('\n'))
            PaperTitle[line['id']] = line['title']
            PaperAbstract[line['id']] = line['abstract_inverted_index']

    if paper_to_test in InformationTheoryCases:
        focal_paper_cits = ThreeBreakthroughs_C[paper_to_test]
        focal_paper_refs= ThreeBreakthroughs_R[paper_to_test]
    else:
        focal_paper_cits = RandomPaper_C[paper_to_test]
        focal_paper_refs = RandomPaper_R[paper_to_test]

    # make a list of all the words in the abstracts of lorenz citations
    content_0_words=[]
    max_num_cit=10
    cit_found=0
    print("looking abstracts from the paper's references..")
    for citation in focal_paper_refs:
        print(citation)
        if citation in PaperAbstract:
            content_0_words = content_0_words+list(PaperAbstract[citation].keys())
            cit_found = cit_found + 1
        else:
            print("can't find " + citation)
        if cit_found >= max_num_cit:
            break


    # make a list of all the words in the focal paper abstract
    print("looking at the abstract from the paper itself..")
    content_1_words = []
    content_1_words + list(PaperAbstract[paper_to_test].keys())

    # make a list of all the words in the abstracts of focal paper references
    print("looking at the abstracts of the papers which cite the focal paper..")
    content_2_words=[]
    max_num_cit=10
    cit_found=0
    for citation in focal_paper_cits:
        print(citation)
        if citation in PaperAbstract:
            content_2_words = content_2_words + list(PaperAbstract[citation].keys())
            cit_found = cit_found + 1
        else:
            print("can't find " + citation)
        if cit_found >= max_num_cit:
            break

    # sanity check, this should be zero
    kl_divergence_should_be_zero= calculate_surprise_between_two_word_lists(
        content_0_words, content_0_words)

    #how surprising is content_1 given content_0?
    Novelty_of_abs_1_given_abs_0 = calculate_surprise_between_two_word_lists(content_0_words,
                                                                             content_1_words)

    #how surprising is content_2 given content_1?
    Novelty_of_abs_2_given_abs_1 = calculate_surprise_between_two_word_lists(content_1_words,
                                                                             content_2_words)
    Novelty_of_abs_2_given_abs_0 = calculate_surprise_between_two_word_lists(content_0_words,
                                                                             content_2_words)

    #Concepts of Transience & Resonance from
    # "Individuals, institutions, and innovation in the debates
    #        of the French Revolution" DeDeo paper
    #Note: KL divergence can be calculated severale ways.
    # "Calculate_sematic_surprise_between_two_word_lists" is better but more expensive
    # "calculate_sematic_surprise_between_two_lines" is cheap and goes by pure probability

    # Novelty: surprise of now, given the past  (how much now is not like the past)
    # Transience: surprise of now, given the future (how much now is not like the future)
    Transience_of_abs_1 = calculate_surprise_between_two_word_lists(content_2_words, content_1_words)

    # Resonance is novelty minus transience
    Resonance_of_abs_1 = Novelty_of_abs_1_given_abs_0-Transience_of_abs_1
    print("Resonance_of_abs_1:",Resonance_of_abs_1)

    #Impact. Is Novelty(2|1) < Novelty(2|0) ?
    #If  abs 0 predicts the future just as well as abs 1, then abs 1 didnt have much impact!
    Impact_of_abs_1 = Novelty_of_abs_2_given_abs_0 - Novelty_of_abs_2_given_abs_1
    print("Impact_of_abs_1:",Impact_of_abs_1)


def calculate_surprise_between_two_word_lists(training_words, testing_words, ):

    all_words = list(set(training_words + testing_words))

    epsilon = 1e-10
    p = np.array([testing_words.count(w) for w in all_words], dtype=float)
    p += epsilon
    p /= np.sum(p)  # Normalize to sum to 1

    q = np.array([training_words.count(w) for w in all_words], dtype=float)
    q += epsilon
    q /= np.sum(q)  # Normalize to sum to 1

    # 4. Calculate Kullback–Leibler divergence using scipy
    # KL(p || q): Surprise of expecting line2 but seeing line1
    kl_divergence = scipy.stats.entropy(pk=p, qk=q)
    print(f"KL Divergence (surprise): {kl_divergence:.4f}")
    return kl_divergence


def calculate_sematic_surprise_between_two_lines(testing_line, training_line):

    # 1. Load a pre-trained sentence transformer (Hugging Face)
    model = SentenceTransformer("all-MiniLM-L6-v2")  #

    # 2. Convert text to high-dimensional semantic embeddings
    emb1 = model.encode(testing_line)
    emb2 = model.encode(training_line)

    # 3. Approximate as probability distributions (e.g., using Softmax)
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    p = softmax(emb1)
    q = softmax(emb2)

    # 4. Calculate KL Divergence
    kl_divergence = scipy.stats.entropy(pk=p, qk=q)
    print(f"KL Divergence (Semantic): {kl_divergence:.4f}")


if __name__ == '__main__':
    calculate_surprise()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
