# evaluate.py

import torch
import sacrebleu
from rapidfuzz.distance import Levenshtein
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from preprocess import load_dataset, prepare_data, tensorize
from model import Encoder, Decoder, Seq2Seq

# dataset class (reuse from train.py)
class TranslationDataset(Dataset):
    def __init__(self, df, urdu_vocab, roman_vocab):
        self.df = df
        self.urdu_vocab = urdu_vocab
        self.roman_vocab = roman_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        urdu_text = self.df.iloc[idx]['urdu']
        roman_text = self.df.iloc[idx]['roman']
        src_tensor = tensorize(urdu_text, self.urdu_vocab, "urdu")
        trg_tensor = tensorize(roman_text, self.roman_vocab, "roman")
        return src_tensor, trg_tensor

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=0)
    trg_batch = pad_sequence(trg_batch, padding_value=0)
    return src_batch, trg_batch

# function to translate a single sentence
def translate_sentence(sentence, model, urdu_vocab, roman_vocab, device, max_len=50):
    model.eval()
    tokens = list(sentence)
    src_tensor = tensorize(sentence, urdu_vocab, "urdu").unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        hidden = torch.tanh(model.fc_hidden(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))).unsqueeze(0)
        cell = torch.tanh(model.fc_cell(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1))).unsqueeze(0)

    outputs = []
    input = torch.tensor([roman_vocab["<sos>"]], device=device)

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input, hidden, cell)
        top1 = output.argmax(1).item()
        if top1 == roman_vocab["<eos>"]:
            break
        outputs.append(top1)
        input = torch.tensor([top1], device=device)

    return " ".join([roman_vocab.lookup_token(idx) for idx in outputs])

# evaluation metrics
def compute_metrics(model, df, urdu_vocab, roman_vocab, device, n_samples=100):
    refs, hyps = [], []
    for i in range(min(n_samples, len(df))):
        urdu_text = df.iloc[i]['urdu']
        roman_ref = df.iloc[i]['roman']
        roman_pred = translate_sentence(urdu_text, model, urdu_vocab, roman_vocab, device)
        refs.append([roman_ref])
        hyps.append(roman_pred)

    bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))
    cer = sum(Levenshtein.distance(hyps[i], refs[i][0]) for i in range(len(hyps))) / len(hyps)
    return bleu.score, cer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_dataset("urdu_ghazals_rekhta.csv")
    train_df, val_df, test_df, urdu_vocab, roman_vocab = prepare_data(df)

    INPUT_DIM = len(urdu_vocab)
    OUTPUT_DIM = len(roman_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    ENC_LAYERS = 2
    DEC_LAYERS = 4
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(torch.load("seq2seq_model.pt", map_location=device))

    bleu, cer = compute_metrics(model, test_df, urdu_vocab, roman_vocab, device)
    print(f"BLEU: {bleu:.2f}, CER: {cer:.2f}")

    for i in range(5):
        urdu_text = test_df.iloc[i]['urdu']
        roman_ref = test_df.iloc[i]['roman']
        roman_pred = translate_sentence(urdu_text, model, urdu_vocab, roman_vocab, device)
        print(f"Urdu: {urdu_text}")
        print(f"Reference: {roman_ref}")
        print(f"Prediction: {roman_pred}")
        print()

if __name__ == "__main__":
    main()
