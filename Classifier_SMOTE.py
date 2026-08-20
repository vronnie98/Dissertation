# pip install sentence-transformers scikit-learn pandas numpy imbalanced-learn

# imports
import pandas as pd
import numpy as np
import time
import os

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

print("Imports completed")

# load the data and look at it
df = pd.read_csv("fake_job_postings.csv")
print(df.shape)   # (rows, columns)
df.head()

# check the class balance
print(df["fraudulent"].value_counts())

# merge the four text columns into one
text_cols = ["company_profile", "description", "requirements", "benefits"]
df[text_cols] = df[text_cols].fillna("")           # blanks -> empty string
df["combined_text"] = df[text_cols].agg(" ".join, axis=1)

# load the Sentence Transformer model
local_dir = os.path.join(os.getcwd(), "all-MiniLM-L6-v2")
if not os.path.exists(local_dir):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir=local_dir)

model = SentenceTransformer(local_dir)
print("Model loaded successfully")

# creating embeddings for all the job posts (or load cached ones if already saved)
if os.path.exists("embeddings.npy"):
    embeddings = np.load("embeddings.npy")
    print("Loaded cached embeddings.npy, shape:", embeddings.shape)
else:
    start = time.time()
    embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)
    end = time.time()
    print("Time taken:", (end - start) / 60, "minutes")
    print("Shape:", embeddings.shape)
    np.save("embeddings.npy", embeddings)
    print("Saved embeddings.npy")

# setting up features (xs) and labels (ys)
xs = embeddings
ys = df["fraudulent"].values

# split into training and test sets BEFORE any resampling
# (SMOTE must never see the test set, or evaluation becomes unrealistic)
xs_train, xs_test, ys_train, ys_test = train_test_split(
    xs, ys, test_size=0.20, random_state=42, stratify=ys
)

print("Training set:", xs_train.shape)
print("Test set:", xs_test.shape)
print("Training class balance before SMOTE:", np.bincount(ys_train))

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # set to your actual core count

# apply SMOTE to the training set only
# https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
smote = SMOTE(
    sampling_strategy="auto",  # resample all classes but the majority up to it
    random_state=42,
    k_neighbors=5,
)
xs_train_res, ys_train_res = smote.fit_resample(xs_train, ys_train)

print("Training class balance after SMOTE:", np.bincount(ys_train_res))
print("Resampled training set:", xs_train_res.shape)

# training Random Forest classifier on the SMOTE-resampled training data
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(xs_train_res, ys_train_res)

# checking the accuracy on train (resampled) and test (original, untouched) sets
print("Train accuracy (resampled):", accuracy_score(ys_train_res, rf_clf.predict(xs_train_res)))
print("Test accuracy:", accuracy_score(ys_test, rf_clf.predict(xs_test)))

# testing to see how well the classifier catches fake posts (not just overall accuracy)
predictions = rf_clf.predict(xs_test)
print(confusion_matrix(ys_test, predictions))
print(classification_report(ys_test, predictions, target_names=["genuine", "fraudulent"]))
