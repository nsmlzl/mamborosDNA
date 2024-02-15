# tested with mamba-ssm v1.1.1, causal-conv1d v1.1.1, transformers v4.36.2,
# torch v2.1.2, pytorch-lightning v2.1.3, torchmetrics v1.2.1, pyfaidx 0.7.2.2

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
import math
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
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy

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
        self.max_n_count = 0

        self.gdata = {}
        dtype = np.dtype([('key', 'U25'), ('start', 'int_'), ('end', 'int_')])
        self.entry_ranges = np.empty(len(ds_entries), dtype=dtype)

        # only append entries of dataset
        count = 0
        for idx, k in enumerate(ds_entries):
            npath = numpy_path + k + '.npy'
            assert os.path.exists(npath), \
                "Numpy file of entry {} does not exist".format(k)

            # dtype of key set to U25; expects string <=25 chars
            assert len(k) <= 25, \
                   "Key string length of {} (len {}) exceeds entry_ranges limit; update dtype".format(k, len(k))

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
        self.n_token_id = self.tokenizer.added_tokens_encoder['N']

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

        inpt = targt = None
        # prevent infinite loops with for loop
        max_samplings = 9
        for i in range(max_samplings + 1):
            rnd_idx = self.rnd_gen.randint(0, self.len - 1)
            inpt, targt = self.get_seq(rnd_idx)
            n_count = torch.sum(targt == self.n_token_id).item()
            if n_count <= self.max_n_count:
                break
            #else:
                #print("too many Ns")

        n_count = torch.sum(targt == self.n_token_id).item()
        if n_count > 0:
            print("WARNING: Training input contains {} Ns".format(n_count))
        return inpt, targt

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
            left_bound = right_bound - self.seq_len - 1
            if left_bound < 0:
                left_bound = 0
            tokens = tokens[left_bound:right_bound]
        else:
            # left and right bounds are switched due to reverse orientation
            right_bound = local_idx
            left_bound = right_bound + self.seq_len + 1
            if left_bound > tokens.size:
                left_bound = tokens.size
            tokens = tokens[right_bound:left_bound][::-1]
        assert tokens.any(), "tokens: {}".format(tokens)

        # use complement when reverse
        if rev_compl:
            tokens = complement(tokens, self.tokenizer)
        # print(seq_str, len(seq_str))

        # add padding
        if tokens.size < self.seq_len + 1:
            padding = np.full((self.seq_len + 1 - tokens.size), self.tokenizer.pad_token_id, dtype=np.byte)
            tokens = np.hstack((padding, tokens))

        assert tokens.size <= self.seq_len + 1

        # TODO use CharTensor instead of LongTensor
        inpt = torch.from_numpy(tokens).to(torch.long)[:-1].clone()
        targt = torch.from_numpy(tokens).to(torch.long)[1:].clone()
        assert inpt.numel() == self.seq_len, "expected a inpt tensor with {} elements; got {}".format(self.seq_len, inpt.numel())
        assert targt.numel() == self.seq_len, "expected a targt tensor with {} elements; got {}".format(self.seq_len, targt.numel())

        return inpt, targt


    # validate T2T dataset; check if correct tokens are returned for predefined indices
    def validate_T2T_ds():
        # check T2T training / validation entry split; no entries are in both sets
        s_train_T2T = set(GenomeDataset.training_entries_T2T)
        s_valid_T2T = set(GenomeDataset.validation_entries_T2T)
        common_T2T = s_train_T2T.intersection(s_valid_T2T)
        assert not common_T2T, \
               "T2T training and validation dataset share common entries {}".format(common_T2T)
        T2T_entry_cnt = len(GenomeDataset.training_entries_T2T) + len(GenomeDataset.validation_entries_T2T)
        assert T2T_entry_cnt == 24, \
               "Expected a total of 24 entries for T2T dataset; got {} entries".format(T2T_entry_cnt)
        print("T2T training and validation sets valid.")
        # check yeast dataset
        s_train_yeast = set(GenomeDataset.training_entries_yeast)
        s_valid_yeast = set(GenomeDataset.validation_entries_yeast)
        common_yeast = s_train_yeast.intersection(s_valid_yeast)
        assert not common_yeast, \
               "Yeast training and validation dataset share common entries {}".format(common_yeast)
        yeast_entry_cnt = len(GenomeDataset.training_entries_yeast) + len(GenomeDataset.validation_entries_yeast)
        assert yeast_entry_cnt == 112, \
               "Expected a total of 112 entries for T2T dataset; got {} entries".format(yeast_entry_cnt)
        print("Yeast training and validation sets valid.")


        yeast_train_iter = GenomeIterator(GenomeDataset.numpy_path, GenomeDataset.training_entries_yeast, 42)
        yeast_valid_iter = GenomeIterator(GenomeDataset.numpy_path, GenomeDataset.validation_entries_yeast, 42)
        T2T_train_iter = GenomeIterator(GenomeDataset.numpy_path, GenomeDataset.training_entries_T2T, 42)
        T2T_valid_iter = GenomeIterator(GenomeDataset.numpy_path, GenomeDataset.validation_entries_T2T, 42)

        token_len = 30
        tokenizer = DNATokenizer()

        for giter in [yeast_train_iter, yeast_valid_iter, T2T_train_iter, T2T_valid_iter]:
            giter.config(tokenizer, token_len)
            assert giter.__next__()

        print(T2T_train_iter.entry_ranges)

        def check(idx, expct_inpt, expct_trgt):
            inpt, trgt = T2T_train_iter.get_seq(idx)
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
        check(5, "[PAD]"*25 + "CACCC", "[PAD]"*24 + "CACCCT")
        check(248387322, "AGGGTTAGGGTTAGGGTTAGGGTTAGGGTT", "GGGTTAGGGTTAGGGTTAGGGTTAGGGTTA")
        check(1067810436, "[PAD]"*6 + "CCTAACCCTAACCCTAACCCCTAA", "[PAD]"*5 + "CCTAACCCTAACCCTAACCCCTAAC")
        check(1228377838, "AGGGTTAGGGTTAGGGGTTAGGGTTAGGGT", "GGGTTAGGGTTAGGGGTTAGGGTTAGGGTT")
        check(2786358510, "CCTAACCCTAACCCTAACCCTAACCCTAAC", "CTAACCCTAACCCTAACCCTAACCCTAACC")
        check(2848818478, "TAGGGTTAGGGTTAGGGTTAGGGTTAGGGT", "AGGGTTAGGGTTAGGGTTAGGGTTAGGGTT")
        # reverse complement
        offset = 2848818499
        check((5 + offset), "GGTTAGGGTTAGGGTTAGGGGTTAGGGTTT", "GTTAGGGTTAGGGTTAGGGGTTAGGGTTTA")
        check(248387322 + offset, "[PAD]"*25 + "AACCC", "[PAD]"*24 + "AACCCT")
        check(1067810436 + offset, "GGTTAGGAGGGTTAGGGGATTAGGGTTAGG", "GTTAGGAGGGTTAGGGGATTAGGGTTAGGG")
        check(1228377838 + offset, "[PAD]"*29 + "T", "[PAD]"*28 + "TA")
        check(2786358510 + offset, "GGTTAGGGTTAGGGTTAGGGTTAGGGTTAG", "GTTAGGGTTAGGGTTAGGGTTAGGGTTAGG")
        check(2848818478 + offset, "[PAD]"*10 + "CTAACCCTAACCCTAACCCT", "[PAD]"*9 + "CTAACCCTAACCCTAACCCTA")
        # check if nothing was modified
        check(2848818478, "TAGGGTTAGGGTTAGGGTTAGGGTTAGGGT", "AGGGTTAGGGTTAGGGTTAGGGTTAGGGTT")
        check(5, "[PAD]"*25 + "CACCC", "[PAD]"*24 + "CACCCT")
        check(2848818478 + offset, "[PAD]"*10 + "CTAACCCTAACCCTAACCCT", "[PAD]"*9 + "CTAACCCTAACCCTAACCCTA")
        check((5 + offset), "GGTTAGGGTTAGGGTTAGGGGTTAGGGTTT", "GTTAGGGTTAGGGTTAGGGGTTAGGGTTTA")

        print("Succesfull validation of T2T dataset access!")


class GenomeDataset(torch.utils.data.IterableDataset):
    hg38_url = 'https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCF_000001405.40/download'
    t2t_url = 'https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCF_009914755.1/download'
    yeast_url = 'http://hypervolu.me/~erik/yeast/cerevisiae.pan.fa.gz'

    T2T_path = "dataset/ncbi_dataset/data/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"
    yeast_path = "dataset/cerevisiae.pan.fa"
    numpy_path = "dataset/numpy/"

    training_entries_T2T = ['NC_060925.1', 'NC_060926.1', 'NC_060927.1', 'NC_060928.1', 'NC_060929.1',
                            'NC_060931.1', 'NC_060932.1', 'NC_060933.1', 'NC_060934.1', 'NC_060935.1',
                            'NC_060936.1', 'NC_060937.1', 'NC_060938.1', 'NC_060939.1', 'NC_060941.1',
                            'NC_060942.1', 'NC_060943.1', 'NC_060944.1', 'NC_060945.1', 'NC_060946.1',
                            'NC_060947.1', 'NC_060948.1']
    validation_entries_T2T = ['NC_060930.1', 'NC_060940.1']

    training_entries_yeast = ['S288C#1#chrII', 'S288C#1#chrIII', 'S288C#1#chrIV', 'S288C#1#chrV',
                              'S288C#1#chrVI', 'S288C#1#chrVII', 'S288C#1#chrIX', 'S288C#1#chrX',
                              'S288C#1#chrXI', 'S288C#1#chrXII', 'S288C#1#chrXIII', 'S288C#1#chrXIV',
                              'S288C#1#chrXVI',
                              'DBVPG6765#1#chrI', 'DBVPG6765#1#chrIII', 'DBVPG6765#1#chrIV',
                              'DBVPG6765#1#chrV', 'DBVPG6765#1#chrVI', 'DBVPG6765#1#chrVII',
                              'DBVPG6765#1#chrVIII', 'DBVPG6765#1#chrX', 'DBVPG6765#1#chrXI',
                              'DBVPG6765#1#chrXII', 'DBVPG6765#1#chrXIII', 'DBVPG6765#1#chrXIV',
                              'DBVPG6765#1#chrXV',
                              'UWOPS034614#1#chrI', 'UWOPS034614#1#chrII', 'UWOPS034614#1#chrIV',
                              'UWOPS034614#1#chrV', 'UWOPS034614#1#chrVI', 'UWOPS034614#1#chrVII',
                              'UWOPS034614#1#chrVIII', 'UWOPS034614#1#chrIX', 'UWOPS034614#1#chrXI',
                              'UWOPS034614#1#chrXII', 'UWOPS034614#1#chrXIII', 'UWOPS034614#1#chrXIV',
                              'UWOPS034614#1#chrXV', 'UWOPS034614#1#chrXVI',
                              'Y12#1#chrI', 'Y12#1#chrII', 'Y12#1#chrIII', 'Y12#1#chrV',
                              'Y12#1#chrVI', 'Y12#1#chrVII', 'Y12#1#chrVIII', 'Y12#1#chrIX',
                              'Y12#1#chrX', 'Y12#1#chrXII', 'Y12#1#chrXIII', 'Y12#1#chrXIV',
                              'Y12#1#chrXV', 'Y12#1#chrXVI',
                              'YPS128#1#chrI', 'YPS128#1#chrII', 'YPS128#1#chrIII', 'YPS128#1#chrIV',
                              'YPS128#1#chrVI', 'YPS128#1#chrVII', 'YPS128#1#chrVIII', 'YPS128#1#chrIX',
                              'YPS128#1#chrX', 'YPS128#1#chrXI', 'YPS128#1#chrXIII', 'YPS128#1#chrXIV',
                              'YPS128#1#chrXV', 'YPS128#1#chrXVI',
                              'SK1#1#chrI', 'SK1#1#chrII', 'SK1#1#chrIII', 'SK1#1#chrIV',
                              'SK1#1#chrV', 'SK1#1#chrVII', 'SK1#1#chrVIII', 'SK1#1#chrIX',
                              'SK1#1#chrX', 'SK1#1#chrXI', 'SK1#1#chrXII', 'SK1#1#chrXIV',
                              'SK1#1#chrXV', 'SK1#1#chrXVI',
                              'DBVPG6044#1#chrI', 'DBVPG6044#1#chrII', 'DBVPG6044#1#chrIII',
                              'DBVPG6044#1#chrIV', 'DBVPG6044#1#chrV', 'DBVPG6044#1#chrVI',
                              'DBVPG6044#1#chrVIII', 'DBVPG6044#1#chrIX', 'DBVPG6044#1#chrX',
                              'DBVPG6044#1#chrXI', 'DBVPG6044#1#chrXII', 'DBVPG6044#1#chrXIII',
                              'DBVPG6044#1#chrXV', 'DBVPG6044#1#chrXVI']
    validation_entries_yeast = ['S288C#1#chrI', 'DBVPG6765#1#chrII', 'UWOPS034614#1#chrIII', 'Y12#1#chrIV',
                                'YPS128#1#chrV', 'SK1#1#chrVI', 'DBVPG6044#1#chrVII', 'S288C#1#chrVIII',
                                'DBVPG6765#1#chrIX', 'UWOPS034614#1#chrX', 'Y12#1#chrXI', 'YPS128#1#chrXII',
                                'SK1#1#chrXIII', 'DBVPG6044#1#chrXIV', 'S288C#1#chrXV', 'DBVPG6765#1#chrXVI']


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

    # manually download Erik's yeast dataset and then convert into numpy arrays
    def get_yeast_data():
        if not Path(GenomeDataset.yeast_path).exists():
            # For now, print instructions for manually downloading the dataset
            print("File not found: {}".format(GenomeDataset.yeast_path))
            print("Download with the following commands:")
            print("$ mkdir dataset; cd dataset")
            print("$ curl -O {}".format(GenomeDataset.yeast_url))
            print("$ gzip -d cerevisiae.pan.fa.gz")
            return

        GenomeDataset.create_np_data(GenomeDataset.yeast_path, GenomeDataset.numpy_path)


# inspired by HyenaDNA and torchmetrics
# https://github.com/HazyResearch/hyena-dna/blob/d553021b483b82980aa4b868b37ec2d4332e198a/src/tasks/torchmetrics.py#L24-L73
# https://github.com/Lightning-AI/torchmetrics/blob/a68455afb9041d1d32c1d6546897fee416abdc41/src/torchmetrics/text/perplexity.py#L28-L88
class FastPerplexity(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    total_log_probs: torch.Tensor
    count: torch.Tensor

    def __init__(self):
        super().__init__()
        self.add_state("total_log_probs", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, loss):
        # expect all GPUs to compute loss over same number of elements
        self.total_log_probs += loss.double()
        self.count += 1

    def compute(self):
        return torch.exp(self.total_log_probs / self.count.double())


class LitMambaDNA(L.LightningModule):
    def __init__(self, dataset, n_layer, d_model, seq_len, lr, lr_scheduler_factor, weight_decay, batch_size_train,
                 batch_size_val, gpu_cnt):
        super().__init__()
        self.dataset = dataset
        if self.dataset == "T2T":
            self.training_entries = GenomeDataset.training_entries_T2T
            self.validation_entries = GenomeDataset.validation_entries_T2T
        elif self.dataset == "yeast":
            self.training_entries = GenomeDataset.training_entries_yeast
            self.validation_entries = GenomeDataset.validation_entries_yeast
        else:
            raise ValueError("unknown dataset: {}".format(self.dataset))

        self.tokenizer = DNATokenizer()
        mamba_config = MambaConfig(n_layer=n_layer, d_model=d_model, vocab_size=self.tokenizer.vocab_size,
                                   ssm_cfg={}, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                                   pad_vocab_size_multiple=1)
        self.mambaDNA = MambaLMHeadModel(mamba_config)
        self.loss_fn = nn.CrossEntropyLoss()

        self.seq_len = seq_len
        self.lr = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.weight_decay = weight_decay
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

        self.train_accuracy = MulticlassAccuracy(num_classes=mamba_config.vocab_size, average='micro')
        self.val_accuracy = MulticlassAccuracy(num_classes=mamba_config.vocab_size, average='micro')
        self.train_perplexity = FastPerplexity()
        self.val_perplexity = FastPerplexity()

        self.save_hyperparameters(ignore=['mambaDNA'])

    def forward(self, inpts):
        return self.mambaDNA(inpts).logits

    def train_dataloader(self):
        seed = torch.distributed.get_rank()
        train_iter = GenomeIterator(GenomeDataset.numpy_path, self.training_entries, seed)
        train_ds = GenomeDataset(train_iter)

        train_ds.config(self.tokenizer, self.seq_len)
        return DataLoader(train_ds, batch_size=self.batch_size_train)

    def training_step(self, batch, batch_idx):
        inpts, trgts = batch
        outpts = self(inpts)
        # print("inpts: {}, trgts: {}, outpts: {}".format(inpts.size(), trgts.size(), outpts.size()))
        loss = self.loss_fn(outpts.view(-1, outpts.size(-1)), trgts.view(-1))
        self.log("train_loss", loss.item(), sync_dist=True)

        if math.isnan(loss.double()):
            raise ValueError("Loss NaN")

        self.train_accuracy(outpts.view(-1, outpts.size(-1)), trgts.view(-1))
        self.train_perplexity(loss)
        return loss

    def on_train_epoch_end(self):
        # log and reset at end of step
        self.log("train_accuracy", self.train_accuracy.compute()*100.0, prog_bar=True)
        self.train_accuracy.reset()
        self.log("train_perplexity", self.train_perplexity.compute(), prog_bar=True)
        self.train_perplexity.reset()

    def val_dataloader(self):
        seed = torch.distributed.get_rank()
        val_iter = GenomeIterator(GenomeDataset.numpy_path, self.validation_entries, seed)
        val_ds = GenomeDataset(val_iter)

        val_ds.config(self.tokenizer, self.seq_len)
        return DataLoader(val_ds, batch_size=self.batch_size_val)

    def validation_step(self, batch, batch_idx):
        inpts, trgts = batch
        outpts = self(inpts)
        loss = self.loss_fn(outpts.view(-1, outpts.size(-1)), trgts.view(-1))

        self.val_accuracy(outpts.view(-1, outpts.size(-1)), trgts.view(-1))
        self.val_perplexity(loss)

    def on_validation_epoch_end(self):
        # log and reset at end of step
        self.log("val_accuracy", self.val_accuracy.compute()*100.0)
        self.val_accuracy.reset()
        self.log("val_perplexity", self.val_perplexity.compute())
        self.val_perplexity.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.mambaDNA.parameters(), lr=self.lr, betas=(0.9, 0.95),
                                      weight_decay=self.weight_decay) #eps=epsilon,
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=150,
                                                                  factor=self.lr_scheduler_factor, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}


def mamba_training(args):
    ckpt_path = args.ckpt_path
    dataset = args.dataset

    # reproducing sec 4.3.2 with 1.3-1.4M parameters, 330B token pretraining
    # model parameters
    n_layer = 16
    d_model = 256

    # training
    gpu_cnt = 8
    max_epochs = 40 * 50
    limit_train_batches = 16
    limit_val_batches = 32

    seq_len = 1024
    batch_size_train = 64
    batch_size_val = 64
    #batches_per_step = 16
    #n_steps = 20000
    #dropout = 0             # original Mamba did not use dropout

    # optimizer
    lr = 8e-3
    lr_scheduler_factor = 0.5
    weight_decay = 0.1
    #epsilon = 0.2 # ???


    torch.set_float32_matmul_precision('medium')

    if ckpt_path == None:
        mambaDNA = LitMambaDNA(dataset, n_layer, d_model, seq_len, lr, lr_scheduler_factor, weight_decay,
                               batch_size_train, batch_size_val, gpu_cnt)
    else:
        print("Loading mambaDNA from checkpoint {}".format(ckpt_path))
        mambaDNA = LitMambaDNA.load_from_checkpoint(ckpt_path, map_location="cpu")

    logger = TensorBoardLogger("tb_logs", name="mamba_model")
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
    trainer = L.Trainer(max_epochs=max_epochs, limit_train_batches=limit_train_batches,
                        limit_val_batches=limit_val_batches, check_val_every_n_epoch=5, gradient_clip_val=1.0,
                        gradient_clip_algorithm="value", devices=gpu_cnt, accelerator="gpu",
                        precision='bf16-mixed', log_every_n_steps=1, logger=logger, strategy="ddp",
                        use_distributed_sampler=False, callbacks=[lr_monitor]) #, profiler='simple')
    trainer.fit(mambaDNA)


def main(args):
    if args.subcommand=='initialize' and args.dataset=='T2T':
        GenomeDataset.get_t2t_data()
    elif args.subcommand=='initialize' and args.dataset=='yeast':
        GenomeDataset.get_yeast_data()
    elif args.subcommand=='validate' and args.module=='dataset':
        GenomeIterator.validate_T2T_ds()
    elif args.subcommand=='validate' and args.module=='tokenizer':
        DNATokenizer.validate()
    elif args.subcommand=='train':
        mamba_training(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="MambaDNA")
    subparsers = parser.add_subparsers(dest='subcommand', required=True)

    initialize_sp = subparsers.add_parser("initialize", help="initialize datasets")
    initialize_sp.add_argument("dataset", choices=['T2T', 'yeast'], help="dataset to initialize")

    validate_sp = subparsers.add_parser("validate", help="validate submodules")
    validate_sp.add_argument("module", choices=['dataset', 'tokenizer'], help="module to validate")

    train_sp = subparsers.add_parser("train", help="run training")
    train_sp.add_argument("--ckpt_path", default=None, metavar="path/to/checkpoint.chpt",
                          help="provide optional checkpoint file to load model from")
    train_sp.add_argument("--dataset", choices=['T2T', 'yeast'], default='T2T',
                          help="select training dataset")
    args = parser.parse_args()

    main(args)
