import sys
import argparse
import os

import numpy as np
import pandas as pd
import nltk

from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from scipy.sparse import vstack

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning



DATASET_NAMES = ['nusa_kalimat', 'nusa_alinea']
DATASET_TO_TASKS = {
    'nusa_kalimat': ['emot', 'senti'],
    'nusa_alinea': ['emot', 'paragraph', 'topic', 'author']
}

DATASET_TO_LANGS = {
    'nusa_kalimat': ['abs', 'btk', 'bew', 'bhp', 'jav', 'mad', 'mak', 'min', 'mui', 'rej', 'sun'],
    'nusa_alinea': ['btk', 'bew', 'bug', 'jav', 'mad', 'mak', 'min', 'mui', 'rej', 'sun']
}

FEAT_COLUMNS = {
    "nusa_kalimat":"text", 
    "nusa_alinea":"paragraph", 
}

LABEL_COLUMNS = {
    "emot":"label", 
    "author":"author_id", 
    "topic":"label",
    "senti":"label",
    "paragraph":"label"
}

def load_data(filedir, feat_col, label_col):
    df = pd.read_csv(filedir)
    data = list(df[feat_col])
    data = [" ".join(word_tokenize(sent)) for sent in data]
    return (data, list(df[label_col]))

def hyperparam_tuning(xtrain, ytrain, xvalid, yvalid, classifier, param_grid):
    # combine train and valid
    x = vstack([xtrain, xvalid])
    y = ytrain + yvalid

    # create predefined split
    # -1 for all training and 0 for all validation
    ps = PredefinedSplit([-1] * len(ytrain) + [0] * len(yvalid))
    clf = GridSearchCV(classifier, param_grid, cv=ps, n_jobs=-1)
    clf = clf.fit(x, y)

    return clf

@ignore_warnings(category=ConvergenceWarning)
def train_and_test(datase_name, task, lang, feat_cols, label_col, directory="../data", feature="BoW"):
    '''
    This function trains and tests machine learning classifiers on a given dataset. It takes in the following parameters:

    dataset_name (str): Name of the dataset to be used.
    task (str): Type of task to be performed on the dataset (e.g., classification, regression).
    lang (str): Language of the dataset.
    feat_col (str): Name of the feature column in the dataset.
    label_col (str): Name of the label column in the dataset.
    directory (str): Directory where the dataset is located (default is "./data").
    feature (str): Type of feature to use for training the classifiers. Only "BoW" and "tfidf" are supported (default is "BoW").
    
    The function first loads the dataset from three CSV files (train, validation, and test), using the specified feature and label columns. It then trains the feature
    extractor on the training data, transforms all the datasets using the fitted feature extractor, and trains and tunes several classifiers using grid search with cross
    validation. Finally, it evaluates the performance of the classifiers on the test dataset using the F1 score, and returns a dictionary of the results.
    '''
    print(f"\tLoading Dataset")
    
    try:
        xtrain, ytrain = load_data(f"{directory}/{dataset_name}-{task}-{lang}-train.csv", feat_cols["nusa_kalimat"], label_col)
        xvalid, yvalid = load_data(f"{directory}/{dataset_name}-{task}-{lang}-valid.csv", feat_cols["nusa_kalimat"], label_col)
        xtest, ytest = load_data(f"{directory}/{dataset_name}-{task}-{lang}-test.csv", feat_cols["nusa_kalimat"], label_col)
    except:
        xtrain, ytrain = load_data(f"{directory}/{dataset_name}-{task}-{lang}-train.csv", feat_cols["nusa_alinea"], label_col)
        xvalid, yvalid = load_data(f"{directory}/{dataset_name}-{task}-{lang}-valid.csv", feat_cols["nusa_alinea"], label_col)
        xtest, ytest = load_data(f"{directory}/{dataset_name}-{task}-{lang}-test.csv", feat_cols["nusa_alinea"], label_col)
    
    print(f"\tlen({directory}/{dataset_name}-{task}-{lang}-train.csv): {len(xtrain)}")
    print(f"\tlen({directory}/{dataset_name}-{task}-{lang}-valid.csv): {len(xvalid)}")
    print(f"\tlen({directory}/{dataset_name}-{task}-{lang}-test.csv): {len(xtest)}")
    
    # train feature on train data
    if feature == "BoW":
        vectorizer = CountVectorizer()
    elif feature == "tfidf":
        vectorizer = TfidfVectorizer()
    else:
        raise Exception('Vectorizer unknown. Use "BoW" or "tfidf"')
    vectorizer.fit(xtrain)

    # transform
    xtrain = vectorizer.transform(xtrain)
    xvalid = vectorizer.transform(xvalid)
    xtest = vectorizer.transform(xtest)
    
    # all classifiers
    classifiers = {"nb" : MultinomialNB(),
                   "svm": SVC(),
                   "lr" : LogisticRegression(),
                  }
    # all params for grid-search
    param_grids = {"nb" : {"alpha": np.linspace(0.001,1,50)},
                   "svm": {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear']},
                   "lr" : {'C': [0.01, 0.1, 1, 10, 100]},
                  }
    
    results = {}
    for c in classifiers:
        print(f"\tTraining and Tuning {c} with feature {feature}")
        clf = hyperparam_tuning(xtrain, ytrain, xvalid, yvalid,
                                classifier=classifiers[c],
                                param_grid=param_grids[c])
        
        print(f"\tEvaluating {c} with feature {feature}")
        pred = clf.predict(xtest.toarray())
        f1score = f1_score(ytest,pred, average='macro')
        results[c] = f1score

    return results

if __name__ == "__main__":
    # Define arguments parser
    nltk.download('punkt')
    parser = argparse.ArgumentParser(description='Evaluate datasets')
    parser.add_argument('--dataset_name', type=str, default='all', help='Name of the dataset to evaluate')
    parser.add_argument('--tasks', type=str, default='all', help='Comma-separated list of tasks to evaluate')
    parser.add_argument('--langs', type=str, default='all', help='Comma-separated list of languages to evaluate')
    args = parser.parse_args()

    results = {}
    # Loop through datasets
    dataset_names = DATASET_NAMES if args.dataset_name == 'all' else [args.dataset_name]
    for dataset_name in dataset_names:
        assert dataset_name in DATASET_NAMES, f"Dataset name '{dataset_name}' not found in DATASET_NAMES"
        # Get tasks to evaluate
        tasks = DATASET_TO_TASKS[dataset_name] if args.tasks == 'all' else args.tasks.split(',')
        for task in tasks:
            assert task in DATASET_TO_TASKS[dataset_name] , f"Task name '{task}' not found in {dataset_name}"
        # Get languages to evaluate
        langs = DATASET_TO_LANGS[dataset_name] if args.langs == 'all' else args.langs.split(',')
        for lang in langs:
            assert lang in DATASET_TO_LANGS[dataset_name], f"Language name '{lang}' not found in {dataset_name}"
        # Loop through tasks and languages
        for task in tasks:
            for lang in langs:
                print(f'{dataset_name}_{task}_{lang}')
                # Loop through all features
                for feature in ['BoW', 'tfidf']:
                    try:
                        # Get the feature and label column for the dataset
                        label_col = LABEL_COLUMNS[task]
                        # Train and test the dataset using the current feature
                        results[f'{dataset_name}_{task}_{lang}_{feature}'] = train_and_test(dataset_name, task, lang, FEAT_COLUMNS,
                                                                                    label_col, feature=feature)
                    except OSError as e:
                        # If unable to open the dataset, print an error message and move on
                        print(f"Unable to open {dataset_name}_{task}_{lang}: {e}", file=sys.stderr)
                        pass

    # Print results for all trained datasets
    for result in results:
        print(f'{result} test metrics:')
        print(results[result])
    
    # Check if the folder exists
    if not os.path.exists('boomer_result'):
        # If it doesn't exist, create it
        os.makedirs('boomer_result')
    

    # Save results to a CSV file
    if args.dataset_name == 'all':
        output_path = "boomer_result/test_results.csv"
    else:
        output_path = f"boomer_result/test_results_{args.dataset_name}_{args.tasks}_{args.langs}.csv"
    df_results = pd.DataFrame.from_dict(results)
    df_results.to_csv(output_path)

    # Print message indicating that evaluation is done
    print("## -- Evaluation Done. -- ##")
