import os
from pathlib import Path
from typing import Any

import numpy as np
from collections import defaultdict
import json
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from sentence_transformers import SentenceTransformer


def calculate_surprise():

    #to do: make everything lower case...?
    #remove strange math symbols

    git_data_dir = Path(__file__).parent
    big_data_dir = "/Users/tdunn/Data/SoS"
    max_num_cit = 30
    surprise_algorithm_to_use=0 #[0=basic distribution, 1=sematic]

    results= {}
    output_path = (
        os.path.join(big_data_dir, "SoS_Resonance_Results.csv"))  # future abstract

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


    papers_to_test=list(InformationTheoryCases.keys()) + list(RandomPaper_C.keys())


    with open(output_path, "w") as file:
        file.write(f"Paper,Type,Resonance,Impactv1,Impactv2\n")

        for paper_to_test in papers_to_test:
            type = "RandomPaper"
            if paper_to_test in InformationTheoryCases:
                focal_paper_cits = ThreeBreakthroughs_C[paper_to_test]
                focal_paper_refs= ThreeBreakthroughs_R[paper_to_test]
                type = "Breakthrough"
            else:
                focal_paper_cits = RandomPaper_C[paper_to_test]
                focal_paper_refs = RandomPaper_R[paper_to_test]

            # make a list of all the words in the abstracts of lorenz citations
            content_0_words=[]
            cit_found=0
            print("looking abstracts from the paper's references..")
            for reffed_paper in focal_paper_refs:
                print(reffed_paper)
                if reffed_paper in PaperAbstract:
                    content_0_words = content_0_words+list(PaperAbstract[reffed_paper].keys())

                    content_0_p_line, content_0_p_words = get_abstract_word_list(
                        PaperAbstract, reffed_paper)
                    content_0_words = content_0_words + content_0_p_words

                    cit_found = cit_found + 1
                else:
                    print("can't find " + reffed_paper)
                if cit_found >= max_num_cit:
                    break
            content_0_line=" ".join(content_0_words)

            # make a list of all the words in the focal paper abstract
            print("looking at the abstract from the paper itself..")
            content_1_line, content_1_words = get_abstract_word_list(PaperAbstract, paper_to_test)

            # make a list of all the words in the abstracts of focal paper references
            print("looking at the abstracts of the papers which cite the focal paper..")
            content_2_words=[]
            cit_found=0
            for cited_paper in focal_paper_cits:
                print(cited_paper)
                if cited_paper in PaperAbstract:
                    content_2_p_line, content_2_p_words = get_abstract_word_list(
                        PaperAbstract, cited_paper)
                    content_2_words = content_2_words + content_2_p_words
                    cit_found = cit_found + 1
                else:
                    print("can't find " + cited_paper)
                if cit_found >= max_num_cit:
                    break
            content_2_line=" ".join(content_2_words)

            # sanity check, this should be zero
            kl_divergence_should_be_zero= calculate_surprise_between_two_word_lists(
                content_0_words, content_0_words)

            #how surprising is content_1 given content_0?
            if surprise_algorithm_to_use==0:
                Novelty_of_abs_1_given_abs_0 = calculate_surprise_between_two_word_lists(content_0_words,
                                                                                     content_1_words)

            #how surprising is content_2 given content_1?
            if surprise_algorithm_to_use==0:
                Novelty_of_abs_2_given_abs_1 = calculate_surprise_between_two_word_lists(content_1_words,
                                                                                         content_2_words)

            if surprise_algorithm_to_use == 0:
                Novelty_of_abs_2_given_abs_0 = calculate_surprise_between_two_word_lists(content_0_words,
                                                                                    content_2_words)
            else:
                Novelty_of_abs_2_given_abs_0 = calculate_sematic_surprise_between_two_lines(
                content_0_line, content_2_line)

            #Concepts of Transience & Resonance from
            # "Individuals, institutions, and innovation in the debates
            #        of the French Revolution" DeDeo paper
            #Note: KL divergence can be calculated severale ways.
            # "calculate_sematic_surprise_between_two_lines" is better but more expensive
            # "calculate_surprise_between_two_word_lists" is cheap and goes by pure probability

            # Novelty: surprise of now, given the past  (how much now is not like the past)
            # Transience: surprise of now, given the future (how much now is not like the future)

            if surprise_algorithm_to_use == 0:
                Transience_of_abs_1 = calculate_surprise_between_two_word_lists(content_2_words, content_1_words)
            else:
                Transience_of_abs_1 = calculate_sematic_surprise_between_two_lines(
                content_2_line, content_1_line)

            # Resonance is novelty minus transience
            Resonance_of_abs_1 = Novelty_of_abs_1_given_abs_0-Transience_of_abs_1
            print("Resonance_of_abs_1:",Resonance_of_abs_1)

            #Impact. Is Novelty(2|1) < Novelty(2|0) ?
            #If  abs 0 predicts the future just as well as abs 1, then abs 1 didnt have much impact!
            Impact_of_abs_1_v1 = Novelty_of_abs_2_given_abs_0 - Novelty_of_abs_2_given_abs_1
            print("Impact_of_abs_1:",Impact_of_abs_1_v1)

            Impact_of_abs_1_v2 = Novelty_of_abs_2_given_abs_0
            print("Impact_of_abs_1_v2:", Impact_of_abs_1_v2)

            results[paper_to_test]=[type, Resonance_of_abs_1, Impact_of_abs_1_v1]
            file.write(f"{paper_to_test},{type},{Resonance_of_abs_1},"
                       f"{Impact_of_abs_1_v1},{Impact_of_abs_1_v2}\n")
            print("\n\n")


def get_abstract_word_list(PaperAbstract: dict[Any, Any], paper_to_test: str) -> tuple[list[Any], LiteralString]:
    content_1_words = []
    allowed_chars = "abcdefghijklmnopqrstuvwxyz"

    for w in PaperAbstract[paper_to_test]:

        word=[[w][0].lower()]
        invalid_chars = set(word) - set(allowed_chars)
        if not invalid_chars:
            content_1_words += word * len(PaperAbstract[paper_to_test][w])

    content_1_line = " ".join(content_1_words)
    return content_1_line, content_1_words


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
    return kl_divergence


if __name__ == '__main__':
    calculate_surprise()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
