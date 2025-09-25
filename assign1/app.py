# app.py

import streamlit as st
import torch
from preprocess import build_vocab
from model import Encoder, Decoder, Seq2Seq
from utils import load_checkpoint, tensor_to_text

# load vocab (must be built same way as in training)
SRC_VOCAB, TRG_VOCAB = build_vocab("data/urdu_roman.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model setup (must match training hyperparams)
INPUT_DIM = len(SRC_VOCAB)
OUTPUT_DIM = len(TRG_VOCAB)
EMB_DIM = 256
HID_DIM = 512
ENC_LAYERS = 2
DEC_LAYERS = 4
DROPOUT = 0.3

encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, DROPOUT)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

# load checkpoint
model = load_checkpoint(model, "checkpoints/best_model.pt", DEVICE)
model.eval()

# translation function
def translate_sentence(sentence, model, src_vocab, trg_vocab, device, max_len=50):
    tokens = ["<sos>"] + list(sentence.strip()) + ["<eos>"]
    src_indexes = [src_vocab[token] if token in src_vocab else src_vocab["<unk>"] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [trg_vocab["<sos>"]]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab["<eos>"]:
            break

    return tensor_to_text(torch.tensor(trg_indexes), trg_vocab)


# streamlit UI
st.title("Urdu → Roman Urdu Translator")
st.write("Type Urdu text below and get Roman Urdu transliteration")

user_input = st.text_area("Enter Urdu text:")

if st.button("Translate"):
    if user_input.strip() != "":
        translation = translate_sentence(user_input, model, SRC_VOCAB, TRG_VOCAB, DEVICE)
        st.success(f"**Roman Urdu:** {translation}")
    else:
        st.warning("Please enter some text to translate.")
