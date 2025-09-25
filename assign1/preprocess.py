# preprocess.py

import re
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator

# load dataset
def load_dataset(path="urdu_ghazals_rekhta.csv"):
    df = pd.read_csv(path)
    df = df[['urdu', 'roman']].dropna().reset_index(drop=True)
    return df

# normalize urdu text
def normalize_urdu(text):
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# normalize roman text
def normalize_roman(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# token generator
def yield_tokens(data_iter, lang):
    for text in data_iter:
        if lang == "urdu":
            yield list(text)  # char-level
        else:
            yield text.split()  # word-level

# prepare data and vocab
def prepare_data(df, test_size=0.25, val_size=0.25):
    df['urdu'] = df['urdu'].apply(normalize_urdu)
    df['roman'] = df['roman'].apply(normalize_roman)

    train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    urdu_vocab = build_vocab_from_iterator(yield_tokens(train_df['urdu'], "urdu"), specials=["<pad>", "<sos>", "<eos>", "<unk>"])
    roman_vocab = build_vocab_from_iterator(yield_tokens(train_df['roman'], "roman"), specials=["<pad>", "<sos>", "<eos>", "<unk>"])

    urdu_vocab.set_default_index(urdu_vocab["<unk>"])
    roman_vocab.set_default_index(roman_vocab["<unk>"])

    return train_df, val_df, test_df, urdu_vocab, roman_vocab

# convert text to tensor ids
def tensorize(text, vocab, lang):
    if lang == "urdu":
        tokens = list(text)
    else:
        tokens = text.split()
    ids = [vocab["<sos>"]] + [vocab[token] for token in tokens] + [vocab["<eos>"]]
    return torch.tensor(ids, dtype=torch.long)
