"""
prepare_emscad.py: cleaning step for my MSc dissertation (topic modelling of job adverts in EMSCAD).

AI assistance: this script was written with help from Claude (an AI assistant made by Anthropic).
See the AI-use statement in the dissertation.

What it does, in order:
  1. load the EMSCAD csv (17,880 job adverts, 866 labelled fraudulent)
  2. clean the text columns (HTML codes, anonymised links/emails/phone numbers, extra spaces)
  3. join the text columns into one text per advert
  4. remove duplicate adverts
  5. save the cleaned data and a log of the counts

It does NOT cluster anything. The cleaning choices follow the pre-processing list from
SECU0057 week 3 (cleaning, duplicates, missing values).

The dataset is not included in this repository. Download "fake_job_postings.csv" from Kaggle
(Real / Fake Job Posting Prediction) and give its path with --input.

Run:
    python src/prepare_emscad.py --input fake_job_postings.csv
"""
import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd

# the five columns that contain advert text (the order matters for the combined text)
TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]

# other columns kept for later (for example to compare the clusters with EMSCAD's own categories)
META_FIELDS = ["location", "department", "salary_range", "telecommuting", "has_company_logo",
               "has_questions", "employment_type", "required_experience", "required_education",
               "industry", "function"]

# EMSCAD replaced real links, emails and phone numbers with long random codes like #URL_ab12...#
# these codes carry no meaning, so they are removed from the text (but counted first)
RE_URL_TOKEN = re.compile(r"#URL_[0-9a-fA-F]+#")
RE_EMAIL_TOKEN = re.compile(r"#EMAIL_[0-9a-fA-F]+#")
RE_PHONE_TOKEN = re.compile(r"#PHONE_[0-9a-fA-F]+#")
# a few adverts also contain normal links and emails
RE_PLAIN_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
RE_PLAIN_EMAIL = re.compile(r"\S+@\S+\.\S+")
# removing a token can leave a bare "http://" behind, so this removes those too
RE_URL_STUB = re.compile(r"https?://", re.I)
# any run of spaces, tabs or new lines
RE_SPACE = re.compile(r"\s+")


def clean_field(text):
    """Clean one text field. Returns (cleaned text, number of links, number of emails, number of phones)."""
    # empty or missing field: nothing to clean
    if not isinstance(text, str) or not text.strip():
        return "", 0, 0, 0

    # turn HTML codes like &amp; into normal characters
    # (done twice because some are double-encoded, e.g. &amp;amp;)
    text = html.unescape(html.unescape(text)).replace("\xa0", " ")

    # count the links, emails and phone numbers before removing them
    n_url = len(RE_URL_TOKEN.findall(text)) + len(RE_PLAIN_URL.findall(text))
    n_email = len(RE_EMAIL_TOKEN.findall(text)) + len(RE_PLAIN_EMAIL.findall(text))
    n_phone = len(RE_PHONE_TOKEN.findall(text))

    # remove them (replace with a space so words next to them do not get glued together)
    for pattern in (RE_URL_TOKEN, RE_EMAIL_TOKEN, RE_PHONE_TOKEN, RE_PLAIN_URL, RE_PLAIN_EMAIL, RE_URL_STUB):
        text = pattern.sub(" ", text)

    # collapse repeated spaces and new lines into one space
    text = RE_SPACE.sub(" ", text).strip()
    return text, n_url, n_email, n_phone


def class_counts(data):
    """Number of adverts in total and in each class (0 = legitimate, 1 = fraudulent)."""
    counts = data["fraudulent"].value_counts().to_dict()
    return {"total": int(len(data)),
            "legitimate": int(counts.get(0, 0)),
            "fraudulent": int(counts.get(1, 0))}


def main():
    # command line options: where the csv is and where to save the results
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="fake_job_postings.csv")
    parser.add_argument("--out", default="data/processed/emscad_clean.parquet")
    parser.add_argument("--log", default="results/prep_log.json")
    args = parser.parse_args()

    # ---- 1. load the data ----------------------------------------------------------------------
    df = pd.read_csv(args.input)
    log = {"input_file": args.input, "raw": class_counts(df)}

    # "duplicate" can mean different things, so count it in several ways to report them
    raw_text = df[TEXT_FIELDS].fillna("").agg(" ".join, axis=1)
    log["duplicate_definitions"] = {
        "all_columns_except_job_id": int(df.drop(columns="job_id").duplicated().sum()),
        "description_only": int(df["description"].duplicated().sum()),
        "title_and_description": int(df.duplicated(["title", "description"]).sum()),
        "combined_raw_text": int(raw_text.duplicated().sum()),
    }

    # ---- 2. clean the text columns -------------------------------------------------------------
    out = df[["job_id", "fraudulent"] + META_FIELDS].copy()
    n_url = pd.Series(0, index=df.index)      # running totals of the removed links, emails, phones
    n_email = pd.Series(0, index=df.index)
    n_phone = pd.Series(0, index=df.index)
    for column in TEXT_FIELDS:
        cleaned = df[column].map(clean_field)           # one (text, links, emails, phones) tuple per advert
        out[column] = cleaned.map(lambda x: x[0])
        n_url += cleaned.map(lambda x: x[1])
        n_email += cleaned.map(lambda x: x[2])
        n_phone += cleaned.map(lambda x: x[3])
    # these counts are only for description, they are not used for the clustering
    out["n_url"] = n_url
    out["n_email"] = n_email
    out["n_phone"] = n_phone

    # ---- 3. empty fields and combined text -----------------------------------------------------
    # 1 if the advert has this field, 0 if it is empty
    # (the company profile is empty much more often in fraudulent adverts, so it is worth keeping)
    for column in ["company_profile", "requirements", "benefits"]:
        out[f"has_{column}"] = (out[column] != "").astype(int)

    # one text per advert: all five fields joined in a fixed order
    out["text_all"] = out[TEXT_FIELDS].agg(" ".join, axis=1).map(lambda s: RE_SPACE.sub(" ", s).strip())
    out["n_words"] = out["text_all"].str.split().str.len()

    # percentage of empty fields in each class
    log["missing_field_pct_by_class"] = {}
    for column in TEXT_FIELDS:
        empty = out[column] == ""
        log["missing_field_pct_by_class"][column] = {
            "legitimate": round(float(empty[out.fraudulent == 0].mean() * 100), 1),
            "fraudulent": round(float(empty[out.fraudulent == 1].mean() * 100), 1)}

    # percentage of adverts that had at least one link, email or phone number removed
    removed_any = (out[["n_url", "n_email", "n_phone"]].sum(axis=1)) > 0
    log["adverts_with_removed_tokens_pct"] = {
        "legitimate": round(float(removed_any[out.fraudulent == 0].mean() * 100), 1),
        "fraudulent": round(float(removed_any[out.fraudulent == 1].mean() * 100), 1)}

    log["empty_text_after_cleaning"] = int((out["n_words"] == 0).sum())
    log["under_20_words_before_dedup"] = int((out["n_words"] < 20).sum())

    # ---- 4. remove duplicates ------------------------------------------------------------------
    # two adverts are duplicates if their cleaned text is the same (ignoring upper/lower case)
    key = out["text_all"].str.lower()

    # first check that no group of duplicates has both labels (fraudulent and legitimate);
    # if it did, we could not tell which label is right, so the whole group would be dropped
    labels_per_text = out.groupby(key)["fraudulent"].nunique()
    conflicting_texts = set(labels_per_text[labels_per_text > 1].index)
    is_conflict = key.isin(conflicting_texts)
    log["duplicate_groups_with_conflicting_labels"] = int(len(conflicting_texts))
    log["rows_dropped_for_conflicting_labels"] = int(is_conflict.sum())
    out = out[~is_conflict].copy()

    # keep the first copy of each advert and drop the rest
    key = out["text_all"].str.lower()
    is_duplicate = key.duplicated(keep="first")
    log["duplicates_removed_on_cleaned_text"] = {
        "total": int(is_duplicate.sum()),
        "legitimate": int(((out.fraudulent == 0) & is_duplicate).sum()),
        "fraudulent": int(((out.fraudulent == 1) & is_duplicate).sum())}
    out = out[~is_duplicate].copy()

    # drop adverts with no text at all (none in EMSCAD, this is just a safety check)
    out = out[out["n_words"] > 0].copy()
    out = out.reset_index(drop=True)

    # ---- 5. summary numbers for the Methods section --------------------------------------------
    log["clean"] = class_counts(out)
    log["fraud_share_pct_clean"] = round(100 * log["clean"]["fraudulent"] / log["clean"]["total"], 2)
    log["words_per_advert_clean"] = {
        "median_all": int(out["n_words"].median()),
        "median_legitimate": int(out.loc[out.fraudulent == 0, "n_words"].median()),
        "median_fraudulent": int(out.loc[out.fraudulent == 1, "n_words"].median()),
        "share_over_256_words_pct": round(float((out["n_words"] > 256).mean() * 100), 1),
        "share_over_512_words_pct": round(float((out["n_words"] > 512).mean() * 100), 1)}
    log["under_20_words_after_dedup"] = int((out["n_words"] < 20).sum())

    # ---- 6. save -------------------------------------------------------------------------------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(json.dumps(log, indent=2))
    print(f"\nSaved {len(out)} adverts to {args.out}; log in {args.log}")


if __name__ == "__main__":
    main()
