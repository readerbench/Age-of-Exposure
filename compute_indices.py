import argparse
import heapq
import json
import os
import pickle
from itertools import repeat
from multiprocessing import Pool, Process

import enchant
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import KeyedVectors
from nltk.corpus import wordnet
from scipy.spatial import procrustes
from scipy.stats import linregress
from sklearn.metrics.pairwise import cosine_similarity

en_us = enchant.Dict("en_US")
en_uk = enchant.Dict("en_UK")
MODEL_NAMES = [
    'Level 1.model',
    'Level 2.model',
    'Level 3.model',
    'Level 4.model',
    'Level 5.model',
    'Level 6.model',
    'Level 7.model',
    'Level 8.model',
    'Level 9.model',
    'Level 10.model'
]
BASELINE = MODEL_NAMES[-1]


def load_model(filepath):
    return KeyedVectors.load(filepath)


def keys(model):
    return list(model.wv.vocab.keys())


def check_if_english_word(word):
    return en_us.check(word) or en_uk.check(word)


def get_vector_space(model, words):
    vector_space = np.zeros((len(words), len(model[keys(model)[0]])))
    for i, word in enumerate(words):
        if hasattr(model, 'wv'):
            # Word2Vec
            wv = model.wv
            vector_size = model.vector_size
        else:
            wv = model
            vector_size = model[list(model.keys())[0]].shape

        if word not in wv:
            vector_space[i] = np.zeros(vector_size)
        else:
            vector_space[i] = wv[word]

    return vector_space


def top_k_cosines(word, k, pw_cosines, word_indices):
    top_k = []

    word_idx = word_indices[word]

    for w in word_indices:
        if w == word:
            continue
        idx = word_indices[w]

        cos_sim = float(pw_cosines[word_idx, idx])

        if len(top_k) < k:
            heapq.heappush(top_k, (cos_sim, w))
        else:
            heapq.heappushpop(top_k, (cos_sim, w))

    top_k.sort(key=lambda t: t[0], reverse=True)
    return [(w, c) for c, w in top_k]


def thresholded_cosines(word, pw_cosines, word_indices, thresh=.3):
    word_idx = word_indices[word]
    cosines = 0
    cosine_sum = 0

    for w in word_indices:
        if w == word:
            continue
        idx = word_indices[w]
        cos_sim = float(pw_cosines[word_idx, idx])

        if cos_sim >= thresh:
            cosines += 1
            cosine_sum += cos_sim

    if cosines > 0:
        cosine_sum /= cosines
    return cosines, cosine_sum


def get_intermediate_values(word, mature_vs, rotations, word_indices):
    values = []

    mature_arr = mature_vs[word_indices[BASELINE][word]].reshape((1, -1))

    for model in rotations:
        if model == BASELINE:
            continue

        if word not in word_indices[model]:
            values.append(0)
            continue

        int_arr = rotations[model][word_indices[model][word]].reshape((1, -1))
        cos_sim = float(cosine_similarity(mature_arr, int_arr))
        values.append(cos_sim)

    return values


def inverse_average(word, mature_vs, rotations, word_indices):
    values = get_intermediate_values(word, mature_vs, rotations, word_indices)

    return 1 - np.mean(values)


def inverse_slope(word, mature_vs, rotations, word_indices):
    values = get_intermediate_values(word, mature_vs, rotations, word_indices)

    slope, _, _, _, _ = linregress(np.arange(len(values)), values)
    if slope == 0:
        return (1 << 31) - 1
    return 1 / slope


def consecutive_index_above_threshold(word, mature_vs, rotations, word_indices, thresh=.4):
    values = get_intermediate_values(word, mature_vs, rotations, word_indices)
    for i in range(len(values) - 1):
        if values[i] >= thresh and values[i + 1] >= thresh:
            return i

    return len(values)


def index_above_threshold(word, mature_vs, rotations, word_indices, thresh=.4):
    values = get_intermediate_values(word, mature_vs, rotations, word_indices)
    for i, value in enumerate(values):
        if value >= thresh:
            return i

    return len(values)


def align_models(models, top_words, disparities_file):
    rotations = {}
    rotation_word_indices = {}
    disparities = []

    for model in MODEL_NAMES[:-1]:
        intermediate_model = models[model]
        mature_model = models[BASELINE]

        supplementary_words = list(
            set(keys(intermediate_model)) - set(top_words))
        supplementary_words = list(
            filter(check_if_english_word, supplementary_words))

        vs_intermediate = get_vector_space(
            intermediate_model, top_words + supplementary_words)
        vs_mature = get_vector_space(mature_model, top_words)

        # Add zeros to the mature model to only rotate on top words
        vs_mature = np.vstack([vs_mature, np.zeros(
            (len(supplementary_words), vs_mature.shape[1]))])

        mtx1, mtx2, disparity = procrustes(vs_mature, vs_intermediate)
        rotations[model] = mtx2
        rotation_word_indices[model] = {
            word: index for index, word in enumerate(top_words + supplementary_words)}

        disparities.append(disparity)

    with open(disparities_file, 'wt') as fout:
        json.dump(disparities, fout)

    return rotations, rotation_word_indices


def process_word(word):
    top3 = top_k_cosines(
        word, 3, pw_cosines[BASELINE], rotation_word_indices[BASELINE])

    inv_avg = inverse_average(
        word, mature_vs, rotations, rotation_word_indices)
    avg = 1 - inv_avg
    inv_slope = inverse_slope(
        word, mature_vs, rotations, rotation_word_indices)
    slope = 1 / inv_slope

    idx = index_above_threshold(
        word, mature_vs, rotations, rotation_word_indices, thresh=.4)
    cidxs = [
        consecutive_index_above_threshold(
            word, mature_vs, rotations, rotation_word_indices, thresh=thresh)
        for thresh in [.3, .35, .4, .45, .5, .55, .6, .65, .7]]

    cosines_above_thresh, avg_cosine_above_thresh = thresholded_cosines(
        word, pw_cosines[BASELINE], rotation_word_indices[BASELINE], thresh=0.3)

    intermediate_values = get_intermediate_values(
        word, mature_vs, rotations, rotation_word_indices)

    result = []
    result.append(word)

    for w, c in top3:
        result.append(w)
        result.append(c)

    result.append(float(np.mean([e[1] for e in top3])))
    result.append(inv_avg)
    result.append(1 - inv_avg)
    result.append(inv_slope)
    result.append(1 / inv_slope)
    result.append(idx)
    result.extend(cidxs)
    result.append(cosines_above_thresh)
    result.append(avg_cosine_above_thresh)
    result.extend(intermediate_values)

    return result


def compute_indices(all_words, output_file, workers=4):
    global pw_cosines, mature_vs, rotations, rotation_word_indices

    columns = ['lemmatized word',
               'highest cosine word', 'highest cosine word similarity',
               '2nd highest cosine word', '2nd highest cosine word similarity',
               '3rd highest cosine word', '3rd highest cosine word similarity',
               'average top3 cosine', 'inverse average', 'average',
               'inverse slope', 'slope', 'index above .4 threshold',
               'continuous index above .3 threshold', 'continuous index above .35 threshold',
               'continuous index above .4 threshold', 'continuous index above .45 threshold',
               'continuous index above .5 threshold', 'continuous index above .55 threshold',
               'continuous index above .6 threshold', 'continuous index above .65 threshold',
               'continuous index above .7 threshold', 'number of cosines above .3 threshold',
               'average cosine above .3 threshold',
               *[f'intermediate cosine similarity {i + 1}' for i in range(len(models) - 1)]]
    data = {column: [] for column in columns}

    pool = Pool(processes=workers)
    results = pool.map_async(process_word, all_words)
    results.wait()

    for result in results.get():
        for i, column in enumerate(data):
            data[column].append(result[i])

    df = pd.DataFrame.from_records(data)[columns]
    df.sort_values(by='lemmatized word', inplace=True)
    df.to_csv(output_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_models_dir', '-i', type=str,
                        default='quantile_models_cds_sorted')
    parser.add_argument('--output_file', '-o', type=str,
                        default='quantiles_tasa_coca_cds.csv')
    parser.add_argument('--frequencies_file', '-f', type=str,
                        default='term_frequencies.json')
    parser.add_argument('--disparities_output_file', '-d', type=str,
                        default='disparities_tasa_coca_cds.json')
    parser.add_argument('--workers', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    global pw_cosines, mature_vs, rotations, rotation_word_indices

    args = parse_args()
    WV_MODELS_DIR = args.input_models_dir
    OUTPUT_FILE = args.output_file
    FREQUENCIES_FILE = args.frequencies_file
    DISPARITIES_OUTPUT_FILE = args.disparities_output_file
    workers = args.workers

    models = {model_name: load_model(os.path.join(WV_MODELS_DIR, model_name))
              for model_name in MODEL_NAMES}
    top_words = keys(models[BASELINE])

    with open(FREQUENCIES_FILE, 'rt') as fin:
        frequencies = json.load(fin)

    rotations, rotation_word_indices = align_models(models,
                                                    top_words,
                                                    DISPARITIES_OUTPUT_FILE)

    all_words = keys(models[BASELINE])
    all_words = list(filter(check_if_english_word, all_words))
    all_words.sort()  # lexicographic ordering

    mature_vs = get_vector_space(models[BASELINE], all_words)
    rotation_word_indices[BASELINE] = {
        word: index for index, word in enumerate(all_words)}
    pw_cosines = {BASELINE: cosine_similarity(mature_vs)}

    print("Number of words to analyze:", len(all_words))
    compute_indices(all_words, OUTPUT_FILE, workers=workers)
