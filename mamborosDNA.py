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
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

from pyfaidx import Fasta
#import pynvml


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
        dtype = np.dtype([('key', 'U45'), ('start', 'int_'), ('end', 'int_')])
        self.entry_ranges = np.empty(len(ds_entries), dtype=dtype)

        # only append entries of dataset
        count = 0
        for idx, k in enumerate(ds_entries):
            npath = numpy_path + k + '.npy'
            assert os.path.exists(npath), \
                "Numpy file of entry {} does not exist".format(k)

            # dtype of key set to U45; expects string <=45 chars
            assert len(k) <= 45, \
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
               "Expected a total of 112 entries for yeast dataset; got {} entries".format(yeast_entry_cnt)
        print("Yeast training and validation sets valid.")
        # check MHC dataset
        s_train_mhc = set(GenomeDataset.training_entries_mhc)
        s_valid_mhc = set(GenomeDataset.validation_entries_mhc)
        common_mhc = s_train_mhc.intersection(s_valid_mhc)
        assert not common_mhc, \
               "MHC training and validation dataset share common entries {}".format(common_mhc)
        mhc_entry_cnt = len(GenomeDataset.training_entries_mhc) + len(GenomeDataset.validation_entries_mhc)
        assert mhc_entry_cnt == 126, \
               "Expected a total of 126 entries for MHC dataset; got {} entries".format(mhc_entry_cnt)
        print("MHC training and validation sets valid.")


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
    mhc_path = "dataset/mhc.fasta"
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

    # preliminary mhc training / validation split
    training_entries_mhc = ['chm13#chr6:28385000-33300000', 'grch38#chr6:28510128-33480000',
                            'HG00438#1#h1tg000040l:22870040-27725000', 'HG00438#2#h2tg000042l:22875152-27895000',
                            'HG00621#1#h1tg000020l:22865680-27905000', 'HG00621#2#h2tg000005l:28460000-33394840',
                            'HG00673#1#h1tg000030l:28480000-33309720', 'HG00673#2#h2tg000018l:0-2900000',
                            'HG00673#2#h2tg000031l:10256-1959976', 'HG00733#1#h1tg000070l:28540000-33419448',
                            'HG00733#2#h2tg000060l:26405000-27483925', 'HG00733#2#h2tg000166l:56-551833',
                            'HG00735#1#h1tg000013l:28505000-33529960', 'HG00735#2#h2tg000038l:768-4899592',
                            'HG00735#2#h2tg000088l:1770000-1971600', 'HG00741#1#h1tg000025l:22875800-27809488',
                            'HG00741#2#h2tg000077l:25160112-30145000', 'HG01071#1#h1tg000017l:136-4190000',
                            'HG01071#1#h1tg000093l:6465160-7272928', 'HG01071#2#h2tg000076l:6400680-11360000',
                            'HG01106#1#h1tg000024l:6450944-11410000', 'HG01106#2#h2tg000019l:115520424-120479488',
                            'HG01109#2#h2tg000001l:27065360-32000000', 'HG01123#1#h1tg000057l:25406272-30460000',
                            'HG01123#2#h2tg000050l:28510000-33454752', 'HG01175#1#h1tg000069l:6460022-11224566',
                            'HG01175#1#h1tg000188l:192-200000', 'HG01175#2#h2tg000032l:22865280-27834488',
                            'HG01243#1#h1tg000074l:64-200000', 'HG01243#1#h1tg000117l:640-4729848',
                            'HG01243#2#h2tg000097l:26140000-30954732', 'HG01243#2#h2tg000146l:1775000-1977373',
                            'HG01258#1#h1tg000066l:26680048-31610000', 'HG01258#2#h2tg000011l:25745000-30694616',
                            'HG01358#1#h1tg000008l:6451192-11445000', 'HG01358#2#h2tg000082l:6461192-11315000',
                            'HG01361#2#h2tg000059l:28530000-33574864', 'HG01891#1#h1tg000024l:25471272-30494488',
                            'HG01891#2#h2tg000027l:26625048-31625000', 'HG01928#1#h1tg000020l:26885896-31930000',
                            'HG01928#2#h2tg000017l:28450768-33499832', 'HG01952#1#h1tg000044l:26960936-31975000',
                            'HG01952#2#h2tg000016l:28530000-33559848', 'HG01978#1#h1tg000035l:28455000-33469848',
                            'HG01978#2#h2tg000046l:965008-6035000', 'HG02055#1#h1tg000074l:0-4714592',
                            'HG02055#1#h1tg000107l:208-210000', 'HG02055#2#h2tg000056l:1096-203976',
                            'HG02080#1#h1tg000032l:28520000-33484896', 'HG02080#2#h2tg000002l:22875040-27915000',
                            'HG02109#1#h1tg000124l:0-4694976', 'HG02109#1#h1tg000192l:1720000-1917499',
                            'HG02109#2#h2tg000055l:951176-5910000', 'HG02145#1#h1tg000017l:6460752-11215000',
                            'HG02145#1#h1tg000194l:3445256-3648582', 'HG02145#2#h2tg000005l:5000-4814248',
                            'HG02145#2#h2tg000204l:3425000-3625704', 'HG02148#1#h1tg000076l:22645936-27570000',
                            'HG02148#2#h2tg000021l:28525000-28725704', 'HG02148#2#h2tg000034l:896-4704704',
                            'HG02257#2#h2tg000080l:1780000-6705000', 'HG02486#1#h1tg000005l:25915024-30869744',
                            'HG02486#2#h2tg000026l:28405256-33400000', 'HG02559#1#h1tg000047l:28525000-33565000',
                            'HG02559#2#h2tg000064l:28480000-33429848', 'HG02572#1#h1tg000052l:0-4685000',
                            'HG02572#2#h2tg000201l:1725000-6619888', 'HG02622#1#h1tg000042l:27005496-32145000',
                            'HG02622#2#h2tg000041l:28535000-33549320', 'HG02630#1#h1tg000088l:15385000-20218811',
                            'HG02630#1#h1tg000147l:3440512-3643392', 'HG02630#2#h2tg000015l:26865000-29774952',
                            'HG02630#2#h2tg000058l:22635880-24764972', 'HG02717#1#h1tg000073l:1775000-6724184',
                            'HG02717#2#h2tg000061l:22650152-27715000', 'HG02723#1#h1tg000100l:1415000-6335000',
                            'HG02723#2#h2tg000107l:22645040-27552632', 'HG02818#1#h1tg000026l:0-4814632',
                            'HG02818#1#h1tg000296l:312-195000', 'HG02818#1#h1tg000358l:0-105984',
                            'HG02818#2#h2tg000019l:16426328-18440000', 'HG02818#2#h2tg000045l:15000-2706770',
                            'HG02818#2#h2tg000293l:192-200000', 'HG02886#1#h1tg000006l:22030032-26955000',
                            'HG03098#1#h1tg000086l:22036272-26959022', 'HG03098#1#h1tg000186l:832-200000',
                            'HG03098#2#h2tg000070l:22020000-27025000', 'HG03453#1#h1tg000094l:576-200000',
                            'HG03453#1#h1tg000148l:865432-5669135', 'HG03453#2#h2tg000229l:5192-205000',
                            'HG03453#2#h2tg000232l:0-4689240', 'HG03486#1#h1tg000034l:22865104-27583779',
                            'HG03486#1#h1tg000131l:1730000-1930576', 'HG03486#2#h2tg000002l:0-4724120',
                            'HG03492#1#h1tg000049l:15375152-20121680', 'HG03492#1#h1tg000215l:1736-204488',
                            'HG03492#2#h2tg000060l:27700512-27903776', 'HG03492#2#h2tg000100l:0-4714952',
                            'HG03516#1#h1tg000073l:22631064-27570000', 'HG03516#2#h2tg000003l:28570000-33584976',
                            'HG03516#2#h2tg000202l:24-441470', 'HG03540#1#h1tg000082l:16455432-21490000',
                            'HG03540#2#h2tg000013l:22880432-27770000', 'HG03579#1#h1tg000035l:16450000-21125662',
                            'HG03579#1#h1tg000097l:1770000-1969808', 'HG03579#2#h2tg000002l:512-4779888',
                            'HG03579#2#h2tg000220l:1096-204360', 'NA18906#1#h1tg000017l:27890048-32825000',
                            'NA18906#2#h2tg000020l:22855296-27800000', 'NA20129#1#h1tg000077l:0-4714952',
                            'NA20129#2#h2tg000038l:320-200000', 'NA20129#2#h2tg000054l:22640000-27332928',
                            'NA21309#1#h1tg000026l:640-4749848', 'NA21309#1#h1tg000294l:3475000-3674040',
                            'NA21309#2#h2tg000021l:0-4629880']
    validation_entries_mhc = ['HG00733#2#h2tg000008l:0-3280000', 'HG01109#1#h1tg000084l:26565424-31500000',
                              'HG01361#1#h1tg000109l:6455112-11370000', 'HG02055#2#h2tg000058l:22631008-27275355',
                              'HG02257#1#h1tg000022l:27000136-31924872', 'HG02572#1#h1tg000139l:6455263-6744671',
                              'HG02723#2#h2tg000038l:23475768-23678648', 'HG02886#2#h2tg000003l:25120800-30214744',
                              'HG03486#2#h2tg000110l:1780000-1978654', 'NA20129#1#h1tg000243l:3460640-3664138',
                              'NA21309#2#h2tg000288l:1725000-1926984']


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
            print("Afterwards rerun initialization subcommand.")
            return

        GenomeDataset.create_np_data(GenomeDataset.yeast_path, GenomeDataset.numpy_path)

    def get_mhc_data():
        if not Path(GenomeDataset.mhc_path).exists():
            # For now, print instructions for manually downloading the dataset
            print("File not found: {}".format(GenomeDataset.mhc_path))
            print("Please download MHC FASTA file and store as {}.".format(GenomeDataset.mhc_path))
            print("Afterwards rerun initialization subcommand.")
            return

        GenomeDataset.create_np_data(GenomeDataset.mhc_path, GenomeDataset.numpy_path)

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


class LitMamborosDNA(L.LightningModule):
    def __init__(self, dataset, n_layer, d_model, seq_len, lr, lr_scheduler_factor, weight_decay, batch_size_train,
                 batch_size_val):
        super().__init__()
        self.dataset = dataset
        if self.dataset == "T2T":
            self.training_entries = GenomeDataset.training_entries_T2T
            self.validation_entries = GenomeDataset.validation_entries_T2T
        elif self.dataset == "yeast":
            self.training_entries = GenomeDataset.training_entries_yeast
            self.validation_entries = GenomeDataset.validation_entries_yeast
        elif self.dataset == "MHC":
            self.training_entries = GenomeDataset.training_entries_mhc
            self.validation_entries = GenomeDataset.validation_entries_mhc
        else:
            raise ValueError("unknown dataset: {}".format(self.dataset))

        self.tokenizer = DNATokenizer()
        mamba_config = MambaConfig(n_layer=n_layer, d_model=d_model, vocab_size=self.tokenizer.vocab_size,
                                   ssm_cfg={'layer': 'Mamba2'}, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                                   pad_vocab_size_multiple=1)
        self.mamborosDNA = MambaLMHeadModel(mamba_config)
        self.loss_fn = nn.CrossEntropyLoss()

        self.seq_len = seq_len
        self.lr = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.weight_decay = weight_decay
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

        self.train_accuracy = MulticlassAccuracy(num_classes=mamba_config.vocab_size, average='micro')
        self.val_accuracy = MulticlassAccuracy(num_classes=mamba_config.vocab_size, average='micro')

        self.val_0_20_accuracy = MulticlassAccuracy(num_classes=mamba_config.vocab_size, average='micro')
        self.val_20_40_accuracy = MulticlassAccuracy(num_classes=mamba_config.vocab_size, average='micro')
        self.val_40_60_accuracy = MulticlassAccuracy(num_classes=mamba_config.vocab_size, average='micro')
        self.val_60_80_accuracy = MulticlassAccuracy(num_classes=mamba_config.vocab_size, average='micro')
        self.val_80_100_accuracy = MulticlassAccuracy(num_classes=mamba_config.vocab_size, average='micro')

        self.train_perplexity = FastPerplexity()
        self.val_perplexity = FastPerplexity()

        self.save_hyperparameters(ignore=['mamborosDNA'])

    def forward(self, inpts):
        return self.mamborosDNA(inpts).logits

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

        def comp_partial_accuracy(metric, outpts_tmp, trgts_tmp, lbound, ubound):
            assert 0.0 <= lbound <= 1.0, "expected lower bound (lbound) in range 0.0 to 1.0."
            assert 0.0 <= ubound <= 1.0, "expected upper bound (ubound) in range 0.0 to 1.0."
            assert lbound < ubound, "expected lower bound (lbound) smaller than upper bound (ubound)."
            vocab_size = outpts_tmp.size(-1)
            lbound = int(self.seq_len * lbound)
            ubound = int(self.seq_len * ubound) + 1
            outpts_tmp = outpts_tmp[:, lbound:ubound, :].contiguous()
            trgts_tmp = trgts_tmp[:, lbound:ubound].contiguous()
            metric(outpts_tmp.view(-1, vocab_size), trgts_tmp.view(-1))

        comp_partial_accuracy(self.val_0_20_accuracy, outpts, trgts, 0.0, 0.2)
        comp_partial_accuracy(self.val_20_40_accuracy, outpts, trgts, 0.2, 0.4)
        comp_partial_accuracy(self.val_40_60_accuracy, outpts, trgts, 0.4, 0.6)
        comp_partial_accuracy(self.val_60_80_accuracy, outpts, trgts, 0.6, 0.8)
        comp_partial_accuracy(self.val_80_100_accuracy, outpts, trgts, 0.8, 1.0)

    def on_validation_epoch_end(self):
        # log and reset at end of step
        self.log("val_accuracy", self.val_accuracy.compute()*100.0)
        self.val_accuracy.reset()
        self.log("val_perplexity", self.val_perplexity.compute())
        self.val_perplexity.reset()

        self.log("val_0_20_accuracy", self.val_0_20_accuracy.compute()*100.0)
        self.val_0_20_accuracy.reset()
        self.log("val_20_40_accuracy", self.val_20_40_accuracy.compute()*100.0)
        self.val_20_40_accuracy.reset()
        self.log("val_40_60_accuracy", self.val_40_60_accuracy.compute()*100.0)
        self.val_40_60_accuracy.reset()
        self.log("val_60_80_accuracy", self.val_60_80_accuracy.compute()*100.0)
        self.val_60_80_accuracy.reset()
        self.log("val_80_100_accuracy", self.val_80_100_accuracy.compute()*100.0)
        self.val_80_100_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.mamborosDNA.parameters(), lr=self.lr, betas=(0.9, 0.95),
                                      weight_decay=self.weight_decay) #eps=epsilon,
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=150,
                                                                  factor=self.lr_scheduler_factor, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}


def mamba_training(args):
    ckpt_path = args.ckpt_path
    dataset = args.dataset

    # model parameters
    # # 1.4M parameter model
    # n_layer = 12
    # d_model = 128

    # # 3.5M parameter model
    # n_layer = 14
    # d_model = 192

    # # 7M parameter model
    # n_layer = 16
    # d_model = 256

    # # 19.3M parameter model
    # n_layer = 20
    # d_model = 384

    # Mamba2 compatible
    n_layer = 20
    d_model = 512

    # # 40.7M parameter model
    # n_layer = 24
    # d_model = 512

    # training
    #gpu_cnt = 8
    max_epochs = 150 * 5 #0
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
    lr_scheduler_factor = 0.85
    weight_decay = 0.1
    #epsilon = 0.2 # ???


    torch.set_float32_matmul_precision('medium')

    if ckpt_path == None:
        mamborosDNA = LitMamborosDNA(dataset, n_layer, d_model, seq_len, lr, lr_scheduler_factor, weight_decay,
                               batch_size_train, batch_size_val)
    else:
        print("Loading mamborosDNA from checkpoint {}".format(ckpt_path))
        mamborosDNA = LitMamborosDNA.load_from_checkpoint(ckpt_path, map_location="cpu")

    logger = TensorBoardLogger("tb_logs", name="mamba_model")
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
    ckpt = L.pytorch.callbacks.ModelCheckpoint(save_top_k=10, monitor="train_loss", save_on_train_epoch_end=True,
                                               verbose=True, every_n_epochs=10)

    trainer = L.Trainer(max_epochs=max_epochs, limit_train_batches=limit_train_batches,
                        limit_val_batches=limit_val_batches, check_val_every_n_epoch=5, gradient_clip_val=0.5,
                        gradient_clip_algorithm="norm", num_nodes=10, devices=8, accelerator="gpu",
                        precision='bf16-mixed', log_every_n_steps=1, logger=logger, strategy="ddp",
                        use_distributed_sampler=False, callbacks=[lr_monitor, ckpt],
                        plugins=[SLURMEnvironment(auto_requeue=False)]) #, profiler='simple')
    trainer.fit(mamborosDNA)


def main(args):
    if args.subcommand=='initialize' and args.dataset=='T2T':
        GenomeDataset.get_t2t_data()
    elif args.subcommand=='initialize' and args.dataset=='yeast':
        GenomeDataset.get_yeast_data()
    elif args.subcommand=='initialize' and args.dataset=='MHC':
        GenomeDataset.get_mhc_data()
    elif args.subcommand=='validate' and args.module=='dataset':
        GenomeIterator.validate_T2T_ds()
    elif args.subcommand=='validate' and args.module=='tokenizer':
        DNATokenizer.validate()
    elif args.subcommand=='train':
        mamba_training(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="MamborosDNA")
    subparsers = parser.add_subparsers(dest='subcommand', required=True)

    initialize_sp = subparsers.add_parser("initialize", help="initialize datasets")
    initialize_sp.add_argument("dataset", choices=['T2T', 'yeast', 'MHC'], help="dataset to initialize")

    validate_sp = subparsers.add_parser("validate", help="validate submodules")
    validate_sp.add_argument("module", choices=['dataset', 'tokenizer'], help="module to validate")

    train_sp = subparsers.add_parser("train", help="run training")
    train_sp.add_argument("--ckpt_path", default=None, metavar="path/to/checkpoint.chpt",
                          help="provide optional checkpoint file to load model from")
    train_sp.add_argument("--dataset", choices=['T2T', 'yeast', 'MHC'], default='T2T',
                          help="select training dataset")
    args = parser.parse_args()

    main(args)
