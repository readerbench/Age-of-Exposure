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
AOA_LISTS = ["Bird", "Bristol", "Cortese", "Kuperman", "Shock",
             "Objective AoA (LR) (months)",
             "Objective AoA (75%) (months)"]
TERM_FREQUENCIES_FILE = 'term_frequencies.json'


class FeatureSelection(Enum):
    LEXICAL = 0
    LEXICAL_AND_WORDNET = 1
    ALL = 2
    ALL_WITHOUT_WORDNET = 3
    ONLY_WORD_TRAJECTORY = 4
    ONLY_WORDNET = 5


def lemmatize(word):
    for tok in nlp(word):
        return tok.lemma_


def get_word_features(word, use_wordnet=False, use_lexical=True):
    global term_frequency

    if not use_wordnet:
        return [
            get_no_syllables(word),
            len(word),
            *term_frequency.get(word, [0] * 4)
        ]
    elif not use_lexical:
        return [
            *get_hypnonymy_tree_sizes(word),
            len(wn.synsets(word)),
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


def get_selected_features(data_columns, feature_selection):
    features = []
    if feature_selection == FeatureSelection.ALL:
        features = data_columns + [
            '# syllables', 'avg. hypernym eccentricities',
            'avg. hyponym eccentricities', '# chars', '# synset',
            'term frequency']
    elif feature_selection == FeatureSelection.LEXICAL:
        features = ['# syllables', '# chars', 'term frequency']
    elif feature_selection == FeatureSelection.LEXICAL_AND_WORDNET:
        features = ['# syllables', 'avg. hypernym eccentricities',
                    'avg. hyponym eccentricities', '# chars', '# synset',
                    'term frequency']

    return features


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


def get_data(aoa_scores, indices_file, feature_selection=FeatureSelection.ALL):
    indices = pd.read_csv(indices_file)
    indices.drop(columns='Unnamed: 0', inplace=True)
    indices = partial_vif_analysis(indices)
    data_columns = get_data_columns(indices)

    intermediate_models = len(
        [index for index in data_columns if 'intermediate' in index])
    print(f"Model with {intermediate_models} intermediate steps.")

    features = get_selected_features(data_columns, feature_selection)

    X = []
    y = []
    y_words = []

    for _, row in indices.iterrows():
        word = row['lemmatized word']
        if word not in aoa_scores:
            continue

        if feature_selection == FeatureSelection.ALL:
            X.append(np.array(row[data_columns]).tolist() +
                     get_word_features(word, use_wordnet=True))
        elif feature_selection == FeatureSelection.LEXICAL:
            X.append(get_word_features(word, use_wordnet=False))
        elif feature_selection == FeatureSelection.ALL_WITHOUT_WORDNET:
            X.append(np.array(row[data_columns]).tolist() +
                     get_word_features(word, use_wordnet=False))
        elif feature_selection == FeatureSelection.LEXICAL_AND_WORDNET:
            X.append(get_word_features(word, use_wordnet=True))
        elif feature_selection == FeatureSelection.ONLY_WORD_TRAJECTORY:
            X.append(np.array(row[data_columns]).tolist())
        elif feature_selection == FeatureSelection.ONLY_WORDNET:
            X.append(get_word_features(word, use_wordnet=True, use_lexical=False))
        else:
            raise ValueError('Invalid feature set:', feature_selection)

        y.append(aoa_scores[word])
        y_words.append(word)

    print(f'{len(X)} / {len(aoa_scores)} words are used from kuperman list')
    X = np.nan_to_num(np.array(X))
    y = np.array(y)

    print('Using', X.shape[1], 'features')

    return X, y


def evaluate_model(X, y, make_model):
    cv_mae = -cross_val_score(make_model(), X, y,
                              scoring='neg_mean_absolute_error',
                              cv=KFold(10, shuffle=True))
    print('MAE:', np.mean(cv_mae), 'norm:', np.mean(cv_mae) / np.max(y))
    print('R2:', np.mean(
        cross_val_score(make_model(), X, y, scoring='r2',
                        cv=KFold(10, shuffle=True))))


def evaluate(X, X_no_vif, y):
    print('\tLinear Regression')
    evaluate_model(
        X_no_vif, y,
        lambda: LinearRegression(normalize=True))

    print('\tLasso Lars AIC')
    evaluate_model(
        X_no_vif, y,
        lambda: LassoLarsIC(criterion='aic', normalize=True))

    print('\tLasso Lars BIC')
    evaluate_model(
        X_no_vif, y,
        lambda: LassoLarsIC(criterion='bic', normalize=True))

    print('\tRF')
    evaluate_model(
        X, y,
        lambda: RandomForestRegressor(n_estimators=100, n_jobs=16))

    print('\tSVR')
    evaluate_model(
        X, y,
        lambda: make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)))


def make_scatterplot(X, y, plot_filename):
    train, test = train_test_split(
        list(range(len(X))), test_size=.1, random_state=42)

    train_X, train_y = X[train], y[train]
    test_X, test_y = X[test], y[test]

    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=100, n_jobs=16))
    model.fit(train_X, train_y)

    pred_train = model.predict(train_X)
    pred_test = model.predict(test_X)

    scatterplot(train_y, pred_train,
                test_y, pred_test,
                filepath=plot_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices_file', '-i', type=str,
                        default='quantiles_tasa_coca_cds.csv')
    parser.add_argument('--tf_file', '-tf', type=str,
                        default='models/tasa_coca_cds_linear/term_frequency.json')
    parser.add_argument('--output_name', '-o', type=str,
                        default='results')
    parser.add_argument('--results_directory', '-r', type=str,
                        default='results')
    return parser.parse_args()


def run_on_scores(aoa_scores, aoa_list, indices_file):
    results_dir = os.path.join(args.results_directory, args.output_name)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    sys.stdout = open(os.path.join(
        results_dir, f'{aoa_list}_results.txt'), 'w')

    scores = aoa_scores[aoa_list]

    for feature_selection in [FeatureSelection.LEXICAL,
                              FeatureSelection.LEXICAL_AND_WORDNET,
                              FeatureSelection.ALL_WITHOUT_WORDNET,
                              FeatureSelection.ALL,
                              FeatureSelection.ONLY_WORDNET,
                              FeatureSelection.ONLY_WORD_TRAJECTORY]:
        print('\n------', feature_selection, '\n')

        X, y = get_data(scores, indices_file, feature_selection)
        X_no_vif = calculate_vif_(X)
        evaluate(X, X_no_vif, y)

        scatterplot_file = f'{aoa_list}_{feature_selection}_scatter.png'
        make_scatterplot(X, y, os.path.join(results_dir, scatterplot_file))

    sys.stdout.close()


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


if __name__ == '__main__':
    args = parse_args()

    term_frequency = get_term_frequency(args.tf_file)
    aoa_scores = load_scores(AOA_LISTS)

    for aoa_list in AOA_LISTS:
        run_on_scores(aoa_scores, aoa_list, args.indices_file)
