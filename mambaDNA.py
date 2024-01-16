# tested with mamba-ssm v1.1.1, causal-conv1d v1.1.1, transformers v4.36.2,
# torch v2.1.2, pytorch-lightning v2.1.3, pyfaidx 0.7.2.2

from zipfile import ZipFile
from io import BytesIO
import requests
import os
import sys
from pathlib import Path
import time
import random
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

from pyfaidx import Fasta
import pynvml


# Tokenizer based on HyenaDNA's tokenizer, which is based on
# https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py
# which itself is inspired by transformer's CanineTokenizer.
# Updated for transformers v4.36.2
# Token mapping:
# [BOS]  : 0
# [UNK]  : 1
# [SEP]  : 2
# [PAD]  : 3
# [CLS]  : 4
# [MASK] : 5
# A      : 6
# C      : 7
# G      : 8
# T      : 9
# N      : 10
# [SWAP] : 11 (extra token space used for swap operation in complement operation)
class DNATokenizer(PreTrainedTokenizer):
    def __init__(self, model_max_length=1073741824, **kwargs):
        # default model_max_length of 1Gbp
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        # added swap token to have extra space when performing complement operation
        characters = ['A', 'C', 'G', 'T', 'N', '[SWAP]']

        super().__init__(
            bos_token=bos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side="left",
            **kwargs,
        )
        self.add_tokens(characters)

    @property
    def vocab_size(self) -> int:
        return len(self.added_tokens_encoder)

    def get_vocab(self):
        return self.added_tokens_encoder

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def validate():
        tokenizer = DNATokenizer()

        assert tokenizer.vocab_size == 12, "unexpected vocab_size"

        assert tokenizer.pad_token_id == 3, "unexpected id for pad token"
        assert tokenizer.mask_token_id == 5, "unexpected id for mask token"

        tokenizer_encode_dict = tokenizer.added_tokens_encoder
        assert len(tokenizer_encode_dict) == 12, "unexpected tokenizer_encode_dict"
        assert tokenizer_encode_dict["[BOS]"] == 0, "unexpected id for \"[BOS]\" token"
        assert tokenizer_encode_dict["[UNK]"] == 1, "unexpected id for \"[UNK]\" token"
        assert tokenizer_encode_dict["[SEP]"] == 2, "unexpected id for \"[SEP]\" token"
        assert tokenizer_encode_dict["[PAD]"] == 3, "unexpected id for \"[PAD]\" token"
        assert tokenizer_encode_dict["[CLS]"] == 4, "unexpected id for \"[CLS]\" token"
        assert tokenizer_encode_dict["[MASK]"] == 5, "unexpected id for \"[MASK]\" token"
        assert tokenizer_encode_dict["A"] == 6, "unexpected id for \"A\" token"
        assert tokenizer_encode_dict["C"] == 7, "unexpected id for \"C\" token"
        assert tokenizer_encode_dict["G"] == 8, "unexpected id for \"G\" token"
        assert tokenizer_encode_dict["T"] == 9, "unexpected id for \"T\" token"
        assert tokenizer_encode_dict["N"] == 10, "unexpected id for \"N\" token"
        assert tokenizer_encode_dict["[SWAP]"] == 11, "unexpected id for \"[SWAP]\" token"

        print("Successfull validation of DNATokenizer!")


def complement(tokens, tokenizer):
    a_token = tokenizer.added_tokens_encoder["A"]
    t_token = tokenizer.added_tokens_encoder["T"]
    c_token = tokenizer.added_tokens_encoder["C"]
    g_token = tokenizer.added_tokens_encoder["G"]
    swap_token = tokenizer.added_tokens_encoder["[SWAP]"]

    tokens = tokens.copy()
    assert np.count_nonzero(tokens==swap_token)==0, \
            "input is not allowed to contain any swap tokens"

    # switch A and T
    tokens[tokens==a_token] = swap_token
    tokens[tokens==t_token] = a_token
    tokens[tokens==swap_token] = t_token
    # switch C and G
    tokens[tokens==c_token] = swap_token
    tokens[tokens==g_token] = c_token
    tokens[tokens==swap_token] = g_token
    return tokens


class GenomeIterator:
    def __init__(self, numpy_path, ds_entries, rnd_seed=0):
        self.gdata = {}
        dtype = np.dtype([('key', 'U20'), ('start', 'int_'), ('end', 'int_')])
        self.entry_ranges = np.empty(len(ds_entries), dtype=dtype)

        # only append entries of dataset
        count = 0
        for idx, k in enumerate(ds_entries):
            npath = numpy_path + k + '.npy'
            assert os.path.exists(npath), \
                "Numpy file of entry {} does not exist".format(k)

            tokens = np.load(npath)
            self.gdata[k] = tokens
            seq_len = self.gdata[k].size
            self.entry_ranges[idx] = np.array([(k, count, count + seq_len)], dtype=dtype)
            count = count + seq_len

        # for e in self.entry_ranges:
        #     print(e)

        # first range forward idices, second range reverse complement
        self.len = self.entry_ranges[-1]['end'] * 2

        self.rnd_seed = rnd_seed
        self.rnd_gen = random.Random(self.rnd_seed)

        self.tokenizer = None
        self.seq_len = None

    def config(self, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def reseed(self):
        world_size = torch.distributed.get_world_size()
        if world_size < 1:
            world_size = 1
        self.rnd_seed = self.rnd_seed + world_size
        self.rnd_gen = random.Random(self.rnd_seed)

    def __next__(self):
        # TODO remove tokenizer from GenomeIterator? Only used for token-ids; no performance improvement
        assert self.tokenizer != None, "Tokenizer need to be set; run config() before usage"
        assert self.seq_len != None, "Sequence length need to be set; run config before usage"

        rnd_idx = self.rnd_gen.randint(0, self.len - 1)
        return self.get_seq(rnd_idx)

    def get_seq(self, idx):
        assert idx >= 0
        assert idx < self.len

        # return reverse complement?
        rev_compl = (idx >= self.entry_ranges[-1]['end'])
        idx = idx % self.entry_ranges[-1]['end']

        # locate FASTA entry of global idx
        key = None
        local_idx = -1
        for e in self.entry_ranges:
            if e['start'] <= idx < e['end']:
                key = e['key']
                local_idx = idx - e['start']

        assert key != None
        assert local_idx != -1

        tokens = self.gdata[key][0]
        if not(rev_compl):
            right_bound = local_idx + 1
            left_bound = right_bound - self.seq_len
            if left_bound < 0:
                left_bound = 0
            tokens = tokens[left_bound:right_bound]
        else:
            # left and right bounds are switched due to reverse orientation
            right_bound = local_idx
            left_bound = right_bound + self.seq_len
            if left_bound > tokens.size:
                left_bound = tokens.size
            tokens = tokens[right_bound:left_bound][::-1]
        assert tokens.any()

        # use complement when reverse
        if rev_compl:
            tokens = complement(tokens, self.tokenizer)
        # print(seq_str, len(seq_str))

        # add padding
        if tokens.size < self.seq_len:
            padding = np.full((self.seq_len - tokens.size), self.tokenizer.pad_token_id, dtype=np.byte)
            tokens = np.hstack((padding, tokens))

        assert tokens.size <= self.seq_len

        # TODO use CharTensor instead of LongTensor
        inpt = torch.from_numpy(tokens).to(torch.long).clone()

        # mask
        target = inpt.clone()
        inpt[-1] = self.tokenizer.mask_token_id
        return inpt, target


    # validate T2T dataset; check if correct tokens are returned for predefined indices
    def validate_T2T_ds():
        train_iter = GenomeIterator(GenomeDataset.numpy_path, GenomeDataset.training_entries, 42)

        token_len = 30
        tokenizer = DNATokenizer()

        train_iter.config(tokenizer, token_len)
        print(train_iter.entry_ranges)

        def check(idx, expct_inpt, expct_trgt):
            inpt, trgt = train_iter.get_seq(idx)
            actual_trgt = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(trgt))
            actual_inpt = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inpt))

            # Does not recognize "PAD" and "MASK" token
            # expct_inpt_len = len(tokenizer(expct_inpt, add_special_tokens=False, max_length=token_len,
            #                            truncation=False)["input_ids"])
            # assert expct_inpt_len == token_len, \
            #     "Unexpected token length of expct_inpt ({})".format(expct_inpt_len)

            assert expct_trgt == actual_trgt, \
                                "Target tokens do not match; expected: {} (len {}), actual: {} (len {})".format(expct_trgt, len(expct_trgt), actual_trgt, len(actual_trgt))
            assert expct_inpt == actual_inpt, \
                                "Input tokens do not match; expected: {} (len {}), actual: {} (len {})".format(expct_inpt, len(expct_inpt), actual_inpt, len(actual_inpt))
            print("tokens match!")

        # forward
        check(5, "[PAD]"*24 + "CACCC" + "[MASK]", "[PAD]"*24 + "CACCC" + "T")
        check(248387322, "GGGTTAGGGTTAGGGTTAGGGTTAGGGTT" + "[MASK]", "GGGTTAGGGTTAGGGTTAGGGTTAGGGTT" + "A")
        check(1067810436, "[PAD]"*5 + "CCTAACCCTAACCCTAACCCCTAA" + "[MASK]", "[PAD]"*5 + "CCTAACCCTAACCCTAACCCCTAA" + "C")
        check(1228377838, "GGGTTAGGGTTAGGGGTTAGGGTTAGGGT" + "[MASK]", "GGGTTAGGGTTAGGGGTTAGGGTTAGGGT" + "T")
        check(2786358510, "CTAACCCTAACCCTAACCCTAACCCTAAC" + "[MASK]", "CTAACCCTAACCCTAACCCTAACCCTAAC" + "C")
        check(2848818478, "AGGGTTAGGGTTAGGGTTAGGGTTAGGGT" + "[MASK]", "AGGGTTAGGGTTAGGGTTAGGGTTAGGGT" + "T")
        # reverse complement
        offset = 2848818499
        check((5 + offset), "GTTAGGGTTAGGGTTAGGGGTTAGGGTTT" + "[MASK]", "GTTAGGGTTAGGGTTAGGGGTTAGGGTTT" + "A")
        check(248387322 + offset, "[PAD]"*24 + "AACCC" + "[MASK]", "[PAD]"*24 + "AACCC" + "T")
        check(1067810436 + offset, "GTTAGGAGGGTTAGGGGATTAGGGTTAGG" + "[MASK]", "GTTAGGAGGGTTAGGGGATTAGGGTTAGG" + "G")
        check(1228377838 + offset, "[PAD]"*28 + "T" + "[MASK]", "[PAD]"*28 + "T" + "A")
        check(2786358510 + offset, "GTTAGGGTTAGGGTTAGGGTTAGGGTTAG" + "[MASK]", "GTTAGGGTTAGGGTTAGGGTTAGGGTTAG" + "G")
        check(2848818478 + offset, "[PAD]"*9 + "CTAACCCTAACCCTAACCCT" + "[MASK]", "[PAD]"*9 + "CTAACCCTAACCCTAACCCT" + "A")
        # check if nothing was modified
        check(2848818478, "AGGGTTAGGGTTAGGGTTAGGGTTAGGGT" + "[MASK]", "AGGGTTAGGGTTAGGGTTAGGGTTAGGGT" + "T")
        check(5, "[PAD]"*24 + "CACCC" + "[MASK]", "[PAD]"*24 + "CACCC" + "T")
        check(2848818478 + offset, "[PAD]"*9 + "CTAACCCTAACCCTAACCCT" + "[MASK]", "[PAD]"*9 + "CTAACCCTAACCCTAACCCT" + "A")
        check((5 + offset), "GTTAGGGTTAGGGTTAGGGGTTAGGGTTT" + "[MASK]", "GTTAGGGTTAGGGTTAGGGGTTAGGGTTT" + "A")

        print("Succesfull validation of T2T dataset access!")


class GenomeDataset(torch.utils.data.IterableDataset):
    hg38_url = 'https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCF_000001405.40/download'
    t2t_url = 'https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCF_009914755.1/download'

    T2T_path = "dataset/ncbi_dataset/data/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"
    numpy_path = "dataset/numpy/"
    training_entries = ['NC_060925.1', 'NC_060926.1', 'NC_060927.1', 'NC_060928.1', 'NC_060929.1',
                        'NC_060931.1', 'NC_060932.1', 'NC_060933.1', 'NC_060934.1', 'NC_060935.1',
                        'NC_060936.1', 'NC_060937.1', 'NC_060938.1', 'NC_060939.1', 'NC_060941.1',
                        'NC_060942.1', 'NC_060943.1', 'NC_060944.1', 'NC_060945.1', 'NC_060946.1',
                        'NC_060947.1', 'NC_060948.1']
    validation_entries = ['NC_060930.1', 'NC_060940.1']

    def __init__(self, genomeIterator):
        super().__init__()
        self.genomeIterator = genomeIterator

    def config(self, tokenizer, seq_len):
        self.genomeIterator.config(tokenizer, seq_len)

    def __iter__(self):
        self.genomeIterator.reseed()
        return self.genomeIterator


    def download_genetic_data(dataset_url):
        print("Download started...")
        response = requests.get(dataset_url, params={'include_annotation_type': 'GENOME_FASTA'})
        if response.status_code == 200:
            data_dir_path = 'dataset'
            os.makedirs(data_dir_path, exist_ok=True)
            with BytesIO(response.content) as zip_buffer:
                ZipFile(zip_buffer, 'r').extractall(path=data_dir_path)
            print("Download complete!")

            print("FASTA files:")
            fpaths = list(Path('dataset').rglob('*.fna'))
            for fpath in fpaths:
                print(fpath)

    def create_np_data(fasta_path, np_dir_path):
        assert Path(fasta_path).exists
        fasta = Fasta(fasta_path, one_based_attributes=False)
        os.makedirs(np_dir_path, exist_ok=True)

        tokenizer = DNATokenizer()

        keys = list(fasta.keys())
        keys = keys
        print("Start conversion of FASTA entries to numpy arrays")
        for idx, k in enumerate(keys):
            # capitalize all nucleotides
            seq = str(fasta[k][:]).upper()
            tokens = tokenizer(seq, add_special_tokens=False, padding=False, truncation=False, return_tensors='np')
            tokens = tokens["input_ids"].astype(np.byte)

            file_path = np_dir_path + k + '.npy'
            np.save(file_path, tokens)
            print("{}/{}: Stored entry {} in {}".format(idx+1, len(keys), k, file_path))

    # download data from NCBI server and precompute for numpy dataset
    def get_t2t_data():
        GenomeDataset.download_genetic_data(GenomeDataset.t2t_url)
        GenomeDataset.create_np_data(GenomeDataset.T2T_path, GenomeDataset.numpy_path)


def mamba_training():
    # parameters
    embed_dim = 128
    n_layers = 6
    dropout = 0             # original Mamba did not use dropout
    # training
    # reproducing sec 4.3.2 with 1.3-1.4M parameters, 330B token pretraining
    seq_len = 1024
    batch_size_train = 1024
    batches_per_step = 16
    batch_size_val = 1024
    n_steps = 5 # 20000
    # optimizer
    lr = 8e-3
    # epsilon = 0.2 # ???
    weight_decay = 0.1

    model_state_dir = "models"
    model_state_path = model_state_dir + "/" + "12-28-2023-1"


    # compute next-token prediction accuracy
    def comp_next_token_pred_acc(prediction, target):
        prediction = prediction[:,-1,:]
        target = target[:,-1]
        _, pred_labels = torch.max(prediction, 1)
        corr_pred = (pred_labels == target)
        accuracy = corr_pred.sum().item() / target.size(-1)
        return accuracy


    class LitMambaDNA(L.LightningModule):
        def __init__(self, mamba_model, seq_len):
            super().__init__()
            self.mamba_model = mamba_model
            self.loss_fn = nn.CrossEntropyLoss()
            self.seq_len = seq_len

        def forward(self, inpts):
            return self.mamba_model(inpts).logits

        def train_dataloader(self):
            seed = torch.distributed.get_rank()
            train_iter = GenomeIterator(GenomeDataset.numpy_path, GenomeDataset.training_entries, seed)
            train_ds = GenomeDataset(train_iter)

            tokenizer = DNATokenizer()
            train_ds.config(tokenizer, seq_len)
            return DataLoader(train_ds, batch_size=batch_size_train)

        def training_step(self, batch, batch_idx):
            inpts, trgts = batch
            outpts = self(inpts)
            # print("inpts: {}, trgts: {}, outpts: {}".format(inpts.size(), trgts.size(), outpts.size()))
            loss = self.loss_fn(outpts.view(-1, outpts.size(-1)), trgts.view(-1))
            
            accuracy = comp_next_token_pred_acc(outpts, trgts)
            #print("batch_idx {}: Loss {:.6f}; Masked prediction accuracy {:.4f}%".format(batch_idx, loss.item(), accuracy*100.0))
            self.log("train_loss", loss.item(), sync_dist=True)
            self.log("train_accuracy", accuracy*100.0, sync_dist=True, prog_bar=True)
            return loss

        def val_dataloader(self):
            seed = torch.distributed.get_rank()
            val_iter = GenomeIterator(GenomeDataset.numpy_path, GenomeDataset.validation_entries, seed)
            val_ds = GenomeDataset(val_iter)

            tokenizer = DNATokenizer()
            val_ds.config(tokenizer, seq_len)
            return DataLoader(val_ds, batch_size=batch_size_val)

        def validation_step(self, batch, batch_idx):
            inpts, trgts = batch
            outpts = self(inpts)
            
            accuracy = comp_next_token_pred_acc(outpts, trgts)
            #print("batch_idx {} validataion: Masked prediction accuracy {:.4f}%".format(batch_idx, accuracy*100.0))
            self.log("val_accuracy", accuracy*100.0, sync_dist=True)

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                          weight_decay=weight_decay) #eps=epsilon, 
            return optimizer
        

    torch.set_float32_matmul_precision('medium')

    print("Not implemented: Model load/store")
    tokenizer = DNATokenizer()

    mamba_config = MambaConfig(d_model=embed_dim, n_layer=n_layers, vocab_size=tokenizer.vocab_size,
                               ssm_cfg={}, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                               pad_vocab_size_multiple=8)
    model = MambaLMHeadModel(mamba_config)
    # dummy model
    #model = nn.Sequential(nn.Embedding(tokenizer.vocab_size, embed_dim), nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, tokenizer.vocab_size))

    # number of parameters of model
    print("#{} model parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    mambaDNA = LitMambaDNA(model, seq_len)

    logger = TensorBoardLogger("tb_logs", name="mamba_model")
    trainer = L.Trainer(max_epochs=1, limit_train_batches=1000, limit_val_batches=int(1), check_val_every_n_epoch=None, val_check_interval=5,
                        devices=8, accelerator="gpu", log_every_n_steps=1, logger=logger, strategy="ddp", use_distributed_sampler=False, profiler='simple')
    trainer.fit(mambaDNA)


def main(args):
    if args.get_dataset:
        GenomeDataset.get_t2t_data()
    if args.validate_dataset:
        GenomeIterator.validate_T2T_ds()
    if args.validate_tokenizer:
        DNATokenizer.validate()
    if args.run_training:
        mamba_training();


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="MambaDNA")
    parser.add_argument("--get_dataset", action="store_true", help="download and precompute T2T dataset")
    parser.add_argument("--validate_dataset", action="store_true", help="validate access of T2T dataset")
    parser.add_argument("--validate_tokenizer", action="store_true", help="validate tokenizer")
    parser.add_argument("-r", "--run_training", action="store_true", help="run training")
    args = parser.parse_args()

    main(args)
