# Topic modelling of fraudulent and legitimate job adverts (EMSCAD)

MSc dissertation project, Department of Security and Crime Science, University College London.
Author: Veronica Oprisan

**Status: work in progress.** The data cleaning is finished. The clustering and topic-modelling analysis has not been run yet, so there are no results in this repository.

## Research question

How can NLP-based topic modelling techniques be used to identify and compare thematic patterns in fraudulent and legitimate online job advertisements using EMSCAD?

Fake job adverts are one way criminals reach victims, including people recruited to move stolen money (money mules). This project looks at what fraudulent adverts talk about, compared with legitimate ones, by sorting a large set of adverts into themes and then checking how the fraudulent and legitimate adverts are spread across those themes.

## Data

EMSCAD, the Employment Scam Aegean Dataset (Vidros et al., 2017): 17,880 real job adverts from 2012 to 2014, of which 866 are labelled fraudulent.

**The data is not included in this repository.** Download `fake_job_postings.csv` from Kaggle ("Real / Fake Job Posting Prediction") and check the dataset's licence and terms before using it.

After cleaning and removing duplicate adverts, the Kaggle copy gives 15,867 adverts (15,154 legitimate and 713 fraudulent).

## Planned method

1. Clean the text (this step is done: `src/prepare_emscad.py`).
2. Turn each advert into numbers with TF-IDF and with sentence embeddings.
3. Group the adverts into themes with K-Means and HDBSCAN, with LDA as a comparison.
4. Name the themes from their top words and by reading sample adverts, with a second person checking a sample.
5. Compare how common each theme is among fraudulent and legitimate adverts, with confidence intervals and tests.
6. Check that the themes are stable (different random starts, different text choices, a class-balanced check).

The settings for each step are written down in an analysis plan before any clustering is run, so that the results are not tuned afterwards. The plan is kept outside this repository.

## What is in this repository

| File | What it is |
|---|---|
| `src/prepare_emscad.py` | Cleaning script for the current analysis |
| `src/cluster_emscad.py` | Features (TF-IDF and sentence embeddings), K-Means, HDBSCAN, LDA, the fraudulent-vs-legitimate comparison and the stability checks. Tested on a small sample only; not yet run on the full data. |
| `requirements.txt` | Package versions used |
| `Classifier.py`, `Classifier.ipynb`, `Classifier_SMOTE.py`, `Classifier_SMOTE.ipynb` | Earlier baseline experiments: a fraud classifier using sentence embeddings and a Random Forest. Kept for reference; they are not part of the current analysis. |

## Running the cleaning step

Developed with Python 3.12.

```
pip install -r requirements.txt
python src/prepare_emscad.py --input fake_job_postings.csv
```

It writes `data/processed/emscad_clean.parquet` (the cleaned adverts) and `results/prep_log.json` (counts of what was removed, for the Methods section). It does not change the original file.

## Running the analysis steps

**These steps have not been run on the full data yet.** Run them from the repository folder after the cleaning step. The first command is a safe test on a small random sample; it prints no results.

```
python src/cluster_emscad.py smoke
```

The steps, in order:

```
python src/cluster_emscad.py features     # TF-IDF representation
python src/cluster_emscad.py kselect      # K-Means for k = 5 to 30, with stability and silhouette scores
python src/cluster_emscad.py kmeans       # final model, theme words, fraudulent-vs-legitimate table
python src/cluster_emscad.py hdbscan      # density-based clustering
python src/cluster_emscad.py lda          # LDA comparison
python src/cluster_emscad.py sensitivity  # reruns with other text and preprocessing choices
python src/cluster_emscad.py balanced     # class-balanced check
python src/cluster_emscad.py embed        # optional: sentence embeddings (needs sentence-transformers)
```

If the author's private analysis plan is present, the script refuses to run the real analysis until the plan has a lock date; without the plan file it skips that check.

## Ethics

The project uses only a public dataset and involves no human participants. A departmental ethics exemption was obtained.

## AI assistance

Parts of the code were written with help from Claude (an AI assistant made by Anthropic). The use of AI is declared in the dissertation.

## Reference

Vidros, S., Kolias, C., Kambourakis, G., & Akoglu, L. (2017). Automatic detection of online recruitment frauds: Characteristics, methods, and a public dataset. *Future Internet, 9*(1), 6. https://doi.org/10.3390/fi9010006
