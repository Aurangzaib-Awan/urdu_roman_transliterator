# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from preprocess import load_dataset, prepare_data, tensorize
from model import Encoder, Decoder, Seq2Seq

# custom dataset
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

# collate function for batching
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=0)
    trg_batch = pad_sequence(trg_batch, padding_value=0)
    return src_batch, trg_batch

# training loop
def train_model(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in iterator:
        src, trg = src.to(model.device), trg.to(model.device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# evaluation loop
def evaluate_model(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(model.device), trg.to(model.device)
            output = model(src, trg, 0)  # no teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# main training script
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

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_data = TranslationDataset(train_df, urdu_vocab, roman_vocab)
    val_data = TranslationDataset(val_df, urdu_vocab, roman_vocab)

    train_iterator = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_iterator = DataLoader(val_data, batch_size=32, collate_fn=collate_fn)

    N_EPOCHS = 10
    CLIP = 1

    for epoch in range(N_EPOCHS):
        train_loss = train_model(model, train_iterator, optimizer, criterion, CLIP)
        val_loss = evaluate_model(model, val_iterator, criterion)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.3f}, Val Loss = {val_loss:.3f}")

    torch.save(model.state_dict(), "seq2seq_model.pt")

if __name__ == "__main__":
    main()
