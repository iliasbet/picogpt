#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

# A GPT from scratch. It has no cache, so it is quite slow at
# generating sequences.

import math

import torch

from torch import nn
from torch.nn import functional as F

import json
from pathlib import Path
import numpy as np

######################################################################


class AddPositionalEncoding(nn.Module):
    def __init__(self, len_max):
        super().__init__()
        self.len_max = len_max

    def forward(self, input):
        u = torch.arange(input.size(1), device=input.device)[:, None]
        j = torch.arange(input.size(2), device=input.device)[None, :]
        k = j % 2
        t = u / (self.len_max ** ((j - k) / input.size(2))) + math.pi / 2 * k
        return input + torch.sin(t)


######################################################################


class QKVAttention(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, nb_heads=1, causal=False, dropout=0.0):
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.causal = causal
        self.dropout = dropout

        self.w_q = randw(nb_heads, dim_qk, dim_in)
        self.w_k = randw(nb_heads, dim_qk, dim_in)
        self.w_v = randw(nb_heads, dim_v, dim_in)
        self.w_o = randw(dim_v * nb_heads, dim_in)

    def forward(self, input):
        q = torch.einsum("ntc,hdc->nhtd", input, self.w_q)
        k = torch.einsum("ntc,hdc->nhtd", input, self.w_k)
        v = torch.einsum("ntc,hdc->nhtd", input, self.w_v)

        a = torch.einsum("nhtd,nhsd->nhts", q, k) / math.sqrt(self.w_q.size(1))

        if self.causal:
            t = torch.arange(input.size(1), device=q.device)
            attzero = t[None, None, :, None] < t[None, None, None, :]
            a = a.masked_fill(attzero, float("-inf"))

        a = a.softmax(dim=3)
        a = F.dropout(a, self.dropout, self.training)
        y = torch.einsum("nhts,nhsd->nthd", a, v).flatten(2)

        return y


class TransformerBlock(nn.Module):

    def __init__(self, dim_model, dim_keys, dim_hidden, nb_heads, causal, dropout):
        super().__init__()
        self.att_ln = nn.LayerNorm((dim_model,))
        self.att_mh = QKVAttention(
            dim_in=dim_model,
            dim_qk=dim_keys,
            dim_v=dim_model // nb_heads,
            nb_heads=nb_heads,
            causal=causal,
            dropout=dropout,
        )
        self.ffn_ln = nn.LayerNorm((dim_model,))
        self.ffn_fc1 = nn.Linear(in_features=dim_model, out_features=dim_hidden)
        self.ffn_fc2 = nn.Linear(in_features=dim_hidden, out_features=dim_model)

    def forward(self, input):
        r = input

        x = self.att_ln(r)
        x = self.att_mh(x)
        r = r + x

        x = self.ffn_ln(r)
        x = self.ffn_fc1(x)
        x = F.relu(x)
        x = self.ffn_fc2(x)
        r = r + x

        return r


class PicoGPT(nn.Module):
    def __init__(
        self,
        voc_size,
        dim_model,
        dim_keys,
        dim_hidden,
        nb_heads,
        nb_blocks,
        causal,
        dropout=0.0,
        len_max=1e5,
    ):
        super().__init__()

        self.starter = nn.Sequential(
            nn.Embedding(voc_size, dim_model),
            nn.Dropout(dropout),
            AddPositionalEncoding(len_max),
        )

        self.trunk = nn.Sequential(
            *[
                TransformerBlock(
                    dim_model, dim_keys, dim_hidden, nb_heads, causal, dropout
                )
                for _ in range(nb_blocks)
            ]
        )

        self.readout = nn.Linear(in_features=dim_model, out_features=voc_size)

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.normal_(mean=0, std=2e-2)
                elif isinstance(m, nn.LayerNorm):
                    m.bias.zero_()
                    m.weight.fill_(1.0)

    def forward(self, input):
        x = F.pad(input, (1, -1))
        x = self.starter(x)
        x = self.trunk(x)
        x = self.readout(x)
        return x

    def cross_entropy(self, input):
        x = self(input)
        return F.cross_entropy(x.transpose(1, 2), input)

    def inplace_ar(self, input, t_start):
        for t in range(t_start, input.size(1)):
            output = self(input)[:, t : t + 1, :]
            dist = torch.distributions.categorical.Categorical(logits=output)
            input[:, t : t + 1] = dist.sample()


######################################################################

if __name__ == "__main__":

    import random

    ######################################################################

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    # Create visualization directory
    viz_dir = Path(".picogpt_viz")
    viz_dir.mkdir(exist_ok=True)
    state_file = viz_dir / "model_state.json"
    embeddings_file = viz_dir / "embeddings.npy"
    attention_file = viz_dir / "attention.npy"

    def save_model_state(sample, loss, epoch, embeddings=None, attention=None):
        """Save current model state for visualization"""
        state = {
            'current_sample': sample,
            'loss': loss,
            'epoch': epoch
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)
        
        if embeddings is not None:
            np.save(embeddings_file, embeddings.cpu().numpy())
        if attention is not None:
            np.save(attention_file, attention.cpu().numpy())

    ######################################################################

    # This function should return a single sample. It has to be a string
    # of always the same length, with a ">" somwehere, always at the same
    # position. This one generate a random sequence of letters of length
    # 25 for the prompt and the same in reversed order to predict.

    def generate_data(nb, prompt_len=25):
        l = [
            "".join(
                [random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(prompt_len)]
            )
            for _ in range(nb)
        ]

        return [x + ">" + x[::-1] for x in l]

    nb_train_samples, nb_test_samples = 10000, 1000

    ######################################################################

    data = generate_data(nb_train_samples + nb_test_samples)

    prompt_lens = set([x.find(">") for x in data])

    assert len(prompt_lens) == 1

    prompt_len = next(iter(prompt_lens))

    assert prompt_len >= 0

    all_symbols = set("".join(data))
    char2token = dict([(c, n) for n, c in enumerate(all_symbols)])
    token2char = dict([(n, c) for n, c in enumerate(all_symbols)])
    voc_size = len(all_symbols)

    data = torch.cat([torch.tensor([char2token[c] for c in s])[None, :] for s in data])

    train_input, test_input = data[:nb_train_samples], data[nb_train_samples:]

    ######################################################################
    # Model

    dim, nb_blocks, nb_heads = 128, 4, 4

    model = PicoGPT(
        voc_size=voc_size,
        dim_model=dim,
        dim_keys=dim // nb_heads,
        dim_hidden=dim,
        nb_heads=nb_heads,
        nb_blocks=nb_blocks,
        causal=True,
        dropout=0.1,
    )

    nb_epochs, batch_size = 11, 100

    optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    train_input = train_input.to(device)
    test_input = test_input.to(device)
    model.to(device)

    nb_parameters = sum([p.numel() for p in model.parameters()])

    print(f"nb_parameters {nb_parameters} device {device}")

    ######################################################################
    # Train / test

    try:
        for n_epoch in range(nb_epochs):
            ######################################################################
            # One training epoch

            model.train()
            acc_train_loss = 0.0

            for batch_idx, input in enumerate(train_input.split(batch_size)):
                loss = model.cross_entropy(input)
                acc_train_loss += loss.item() * input.size(0)
                optim.zero_grad()
                loss.backward()
                optim.step()

                # Save state for visualization every 10 batches
                if batch_idx % 10 == 0:
                    with torch.no_grad():
                        # Get sample text
                        sample = "".join([token2char[x.item()] for x in input[0]])
                        
                        # Get embeddings
                        x = F.pad(input[0:1], (1, -1))
                        embeddings = model.starter[0](x)
                        
                        # Get attention from first layer
                        x = model.starter(x)
                        x = model.trunk[0].att_ln(x)
                        q = torch.einsum("ntc,hdc->nhtd", x, model.trunk[0].att_mh.w_q)
                        k = torch.einsum("ntc,hdc->nhtd", x, model.trunk[0].att_mh.w_k)
                        attention = torch.einsum("nhtd,nhsd->nhts", q, k) / np.sqrt(q.size(-1))
                        
                        if model.trunk[0].att_mh.causal:
                            t = torch.arange(x.size(1), device=device)
                            mask = t[:, None] < t[None, :]
                            attention = attention.masked_fill(mask[None, None, :, :], float('-inf'))
                        
                        attention = torch.softmax(attention, dim=-1)
                        attention = attention.mean(dim=1).squeeze(0)
                        
                        save_model_state(sample, loss.item(), n_epoch, embeddings[0], attention)

            acc_train_loss = acc_train_loss / train_input.size(0)

            ######################################################################
            # One test epoch

            model.eval()
            acc_test_loss = 0.0

            for input in test_input.split(batch_size):
                loss = model.cross_entropy(input)
                acc_test_loss += loss.item() * input.size(0)

            acc_test_loss = acc_test_loss / test_input.size(0)

            input = test_input[:batch_size]
            result = input.clone()
            result[:, prompt_len:] = 0
            model.inplace_ar(result, t_start=prompt_len)

            nb_errors = (input[:, prompt_len:] != result[:, prompt_len:]).long().sum().item()
            error_rate = nb_errors / input[:, prompt_len:].numel()

            print(
                f"n_epoch {n_epoch} train_loss {acc_train_loss} test_loss {acc_test_loss} token_error {error_rate*100:.01f}%"
            )

            if n_epoch % 10 == 0:
                print("----------------------------------------------------------------------")
                for s, t in zip(input[:5], result):
                    print("true:      " + "".join([token2char[x.item()] for x in s]))
                    print("generated: " + "".join([token2char[x.item()] for x in t]))
                print("----------------------------------------------------------------------")

    finally:
        # Cleanup visualization files
        if viz_dir.exists():
            for file in viz_dir.glob("*"):
                file.unlink()
            viz_dir.rmdir()
