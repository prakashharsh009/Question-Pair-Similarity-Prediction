import warnings
import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from featurizer import extract_features

warnings.filterwarnings("ignore")


print("Loading Models............")

## Loading Models
d = "./models/"
with open(d + "std_tfidf.pkl", "rb") as f:
    std = pickle.load(f)
with open(d + "tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(d + "tfidf_GBDT_model.pkl", "rb") as f:
    model = pickle.load(f)


def predict(q1, q2, prob):
    a = "me"
    # convert it into dataframe
    new_df = pd.DataFrame(columns=["question1", "question2"])
    new_df = new_df.append({"question1": q1, "question2": q2}, ignore_index=True)
    new_df = extract_features(new_df)  # getting advance and basic features

    # get the tfidf vectorizer of text
    x_q1 = vectorizer.transform(new_df["question1"])
    x_q2 = vectorizer.transform(new_df["question2"])
    cols = [i for i in new_df.columns if i not in ["question1", "question2"]]
    new_df = new_df.loc[:, cols].values

    # get the hand crafted features
    X = hstack((x_q1, x_q2, new_df)).tocsr()
    X = std.transform(X)

    y_q = model.predict(X)
    y_q_proba = model.predict_proba(X)

    return y_q, y_q_proba
