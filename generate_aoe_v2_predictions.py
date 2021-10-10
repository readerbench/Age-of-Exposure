import argparse
import json
import os
import sys
from enum import Enum

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.metrics import confusion_matrix, make_scorer, mean_absolute_error
from sklearn.model_selection import (KFold, cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

from utils import *


nlp = spacy.load('en_core_web_lg')
MODEL_TYPES = ['rf']
AOA_LISTS = ["Kuperman"]


def lemmatize(word):
    for tok in nlp(word):
        return tok.lemma_


def get_term_frequency(tf_file):
    with open(tf_file, 'rt') as fin:
        all_term_frequency = json.load(fin)
        term_frequency = {}
        for k in all_term_frequency[-1]:
            for i in range(len(all_term_frequency)):
                if k not in term_frequency:
                    term_frequency[k] = []

                step_tf = all_term_frequency[i].get(k, 0) / 1_000_000
                term_frequency[k].append(step_tf)

    words = list(term_frequency.keys())
    tf_X = np.array([term_frequency[w] for w in words])
    tf_X_vif = calculate_vif_(tf_X)
    for i, w in enumerate(words):
        term_frequency[w] = tf_X_vif[i].tolist() + [np.mean(tf_X[i]), np.std(tf_X[i])]

    return term_frequency


def get_word_features(word, use_wordnet=False):
    global term_frequency

    if not use_wordnet:
        return [
            get_no_syllables(word),
            len(word),
            *term_frequency.get(word, [0] * 4)
        ]
    else:
        return [
            get_no_syllables(word),
            *get_hypnonymy_tree_sizes(word),
            len(word),
            len(wn.synsets(word)),
            *term_frequency.get(word, [0] * 4)
        ]


def calculate_vif_(X, thresh=5.0, return_variables=False):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped and len(variables) > 1:
        dropped = False
        vif = [variance_inflation_factor(X[:, variables], ix)
               for ix in range(X[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            del variables[maxloc]
            dropped = True

    if not return_variables:
        return X[:, variables]
    else:
        return X[:, variables], variables


def get_data_columns(indices):
    data_columns = indices.columns[1:].tolist()

    data_columns.remove('highest cosine word')
    data_columns.remove('2nd highest cosine word')
    data_columns.remove('3rd highest cosine word')

    return data_columns


def partial_vif_analysis(indices):
    new_indices = indices.copy()
    template = 'continuous index above'
    selected_columns = []
    for col in indices.columns:
        if template not in col:
            continue
        selected_columns.append(col)

    X = np.array(indices[selected_columns])
    _, filtered_col = calculate_vif_(X, return_variables=True)
    print('Continuous indices filtered down to:',
          ', '.join([selected_columns[i] for i in filtered_col]))
    for i, col in enumerate(selected_columns):
        if i not in filtered_col:
            del new_indices[col]

    return new_indices


def get_data(aoa_scores, indices_file):
    indices = pd.read_csv(indices_file)
    indices.drop(columns='Unnamed: 0', inplace=True)
    indices = partial_vif_analysis(indices)
    data_columns = get_data_columns(indices)

    intermediate_models = len(
        [index for index in data_columns if 'intermediate' in index])
    print(f"Model with {intermediate_models} intermediate steps.")

    X = []
    y = []
    
    test_X = []
    test_words = []

    for _, row in indices.iterrows():
        word = row['lemmatized word']

        if not isinstance(word, str):
            continue

        sample = np.array(row[data_columns]).tolist() + get_word_features(word, use_wordnet=True)
        test_X.append(sample)
        test_words.append(word)

        if word not in aoa_scores:
            continue

        X.append(sample)
        y.append(aoa_scores[word])

    print(f'{len(X)} / {len(aoa_scores)} words are used from kuperman list')
    X = np.nan_to_num(np.array(X))
    test_X = np.nan_to_num(np.array(test_X))
    y = np.array(y)

    print('Using', X.shape[1], 'features')

    return X, y, test_X, test_words


def get_aoe_word_predictions(indices_file):
    train_X, train_y, test_X, test_words = get_data(aoa_scores['Kuperman'], indices_file)

    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, n_jobs=16))
    model.fit(train_X, train_y)
    word_predictions = model.predict(test_X)

    word_scores = {'word': test_words, 'AoE': word_predictions.tolist()}
    return pd.DataFrame.from_records(word_scores)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices_file', '-i', type=str,
                        default='indices/indices_tasa_coca_cds_linear.csv')
    parser.add_argument('--term_frequency', '-tf', type=str, required=True)
    parser.add_argument('--output_file', '-o', type=str,
                        default='results/anova_tasa_coca_cds.csv')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    term_frequency = get_term_frequency(args.term_frequency)
    aoa_scores = load_scores(AOA_LISTS)

    predictions = get_aoe_word_predictions(args.indices_file)
    predictions.to_csv(args.output_file, index=None)