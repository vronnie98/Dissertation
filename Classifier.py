# pip install sentence-transformers scikit-learn pandas numpy

# imports
import pandas as pd
import numpy as np
import time

from sentence_transformers import SentenceTransformer   
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

print("Imports completed")

# load the data and look at it
df = pd.read_csv("fake_job_postings.csv")   
print(df.shape)   # (rows, columns)
df.head()

# check the class balance
df["fraudulent"].value_counts()

# merge the four text columns into one
text_cols = ["company_profile", "description", "requirements", "benefits"]
df[text_cols] = df[text_cols].fillna("")           # blanks -> empty string
df["combined_text"] = df[text_cols].agg(" ".join, axis=1)

df["combined_text"].iloc[0]   # look at the first merged post

# load the Sentence Transformer model
#model = SentenceTransformer("all-MiniLM-L6-v2")
#print("Model loaded successfully") #this didn't work so i asked claude to help

# load the Sentence Transformer model
import os
from huggingface_hub import snapshot_download

local_dir = os.path.join(os.getcwd(), "all-MiniLM-L6-v2")
if not os.path.exists(local_dir):
    snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir=local_dir)

model = SentenceTransformer(local_dir)
print("Model loaded successfully")

# testing on a small subset first to check the timing
subset = df["combined_text"].head(100).tolist()

start = time.time()
embeddings = model.encode(subset)
end = time.time()

print("Time taken:", end - start, "seconds")
print("Shape:", embeddings.shape)

# creating embeddings for all the job posts
start = time.time()
embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)
end = time.time()

print("Time taken:", (end - start) / 60, "minutes")
print("Shape:", embeddings.shape)

# save the embeddings so we don't have to redo this every time
np.save("embeddings.npy", embeddings)
print("Saved embeddings.npy")

# seting up features (xs) and labels (ys)
xs = embeddings
ys = df["fraudulent"].values

# split into training and test sets 
xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.20, random_state=42)

print("Training set:", xs_train.shape)
print("Test set:", xs_test.shape)

# training Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(xs_train, ys_train)

# checking the accuracy on train and test sets
print("Train accuracy:", accuracy_score(ys_train, rf_clf.predict(xs_train)))
print("Test accuracy:", accuracy_score(ys_test, rf_clf.predict(xs_test)))

# testing to see how well the classifier catches fake posts (not just overall accuracy)
predictions = rf_clf.predict(xs_test)
print(confusion_matrix(ys_test, predictions))


