import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LinearRegression
import nltk
import syllables
import networkx as nx
from nltk.corpus import wordnet as wn


AOA_FILE = 'AoA_MCE.csv'


def load_scores(metrics):
    df = pd.read_csv(AOA_FILE)
    df = df[metrics + ['Word']]

    scores = {}
    for metric in metrics:
        scores[metric] = {}
        for _, row in df[[metric, 'Word']].dropna().iterrows():
            scores[metric][row['Word'].lower()] = row[metric]

    return scores


def linear_fit(true, pred):
    model = LinearRegression()
    model.fit(pred.reshape(-1, 1), true.reshape(-1, 1))
    fit_line = np.arange(true.min(), true.max() + 1).reshape(-1, 1)

    return model.predict(fit_line)


def scatterplot(y_true_train, y_pred_train,
                y_true_test, y_pred_test,
                xlabel='AoA',
                ylabel='Predicted AoA',
                filepath=None):
    predictions_df_train = pd.DataFrame.from_records({
            xlabel: y_true_train.ravel(),
            ylabel: y_pred_train.ravel(),
            'error': np.abs(y_true_train.ravel() - y_pred_train.ravel()).astype(np.int32)
    })
    predictions_df_test = pd.DataFrame.from_records({
        xlabel: y_true_test.ravel(),
        ylabel: y_pred_test.ravel(),
        'error': np.abs(y_true_test.ravel() - y_pred_test.ravel()).astype(np.int32)
    })

    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(12)
    fig.set_figheight(6)
    fig.set_tight_layout('tight')

    ax[0].plot(np.arange(min(y_true_train), max(y_true_train) + 1),
               np.arange(min(y_true_train), max(y_true_train) + 1), label='ideal fit')
    ax[0].plot(np.arange(min(y_true_train), max(y_true_train) + 1),
               linear_fit(y_true_train, y_pred_train), label='observed linear fit')
    ax[0].set_title('Train')
    ax[1].plot(np.arange(min(y_true_test), max(y_true_test) + 1),
               np.arange(min(y_true_test), max(y_true_test) + 1), label='ideal fit')
    ax[1].plot(np.arange(min(y_true_test), max(y_true_test) + 1),
               linear_fit(y_true_test, y_pred_test), label='observed linear fit')
    ax[1].set_title('Test')
    sns.scatterplot(
        x=xlabel, y=ylabel, data=predictions_df_train, hue='error', size_norm=True, ax=ax[0])
    sns.scatterplot(
        x=xlabel, y=ylabel, data=predictions_df_test, hue='error', size_norm=True, ax=ax[1])
    fig.set_label('Scatterplot of AoA and predicted AoA values')
    
    if filepath:
        fig.savefig(filepath)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    fig.set_figheight(12)
    fig.set_figwidth(12)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def get_no_syllables(word):
    return syllables.estimate(word)


def closure_graph(synset, fn):
    visited = set([synset])
    G = nx.DiGraph()
    S = [synset]
    
    while S != []:
        u = S.pop(-1)
    
        for v in fn(u):
            if v in visited:
                continue
            
            visited.add(v)
            G.add_node(v.name())
            G.add_edge(u.name(), v.name())
            S.append(v)

    return G


def get_hypnonymy_tree_sizes(word):
    hypernym_trees = []
    hyponym_trees = []

    hypernym_eccentricities = []
    hyponym_eccentricities = []

    for synset in wn.synsets(word):
        hypernym_trees.append(closure_graph(synset, lambda s: s.hypernyms()))
        hyponym_trees.append(closure_graph(synset, lambda s: s.hyponyms()))

        hypernym_tree = hypernym_trees[-1]
        hyponym_tree = hyponym_trees[-1]

        if len(hypernym_tree) != 0:
            hypernym_eccentricities.append(
                nx.eccentricity(hypernym_tree, v=synset.name()))

        if len(hyponym_tree) != 0:
            hyponym_eccentricities.append(
                nx.eccentricity(hyponym_tree, v=synset.name()))
 
    return np.mean(hypernym_eccentricities), np.mean(hyponym_eccentricities)
