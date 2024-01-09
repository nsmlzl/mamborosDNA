# tested with mamba-ssm v1.1.1, causal-conv1d v1.1.1, transformers v4.36.2,
# torch v2.1.2, pytorch-lightning v2.1.3, pyfaidx 0.7.2.2

from zipfile import ZipFile
from io import BytesIO
import requests
import os
from pathlib import Path
import time
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from pyfaidx import Fasta
import pynvml



# HyenaDNA tokenizer; code from their jupyter notebook
"""
Just a simple character level tokenizer.

From: https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py

CharacterTokenzier for Hugging Face Transformers.
This is heavily inspired from CanineTokenizer in transformers package.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


def mamba_training():
    class CharacterTokenizer(PreTrainedTokenizer):
        def __init__(self, characters: Sequence[str], model_max_length: int, padding_side: str='left', **kwargs):
            """Character tokenizer for Hugging Face transformers.
            Args:
                characters (Sequence[str]): List of desired characters. Any character which
                    is not included in this list will be replaced by a special token called
                    [UNK] with id=6. Following are list of all of the special tokens with
                    their corresponding ids:
                        "[CLS]": 0
                        "[SEP]": 1
                        "[BOS]": 2
                        "[MASK]": 3
                        "[PAD]": 4
                        "[RESERVED]": 5
                        "[UNK]": 6
                    an id (starting at 7) will be assigned to each character.
                model_max_length (int): Model maximum sequence length.
            """
            self.characters = characters
            self.model_max_length = model_max_length
            bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
            eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
            sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
            cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
            pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
            unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

            mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

            self._vocab_str_to_int = {
                "[CLS]": 0,
                "[SEP]": 1,
                "[BOS]": 2,
                "[MASK]": 3,
                "[PAD]": 4,
                "[RESERVED]": 5,
                "[UNK]": 6,
                **{ch: i + 7 for i, ch in enumerate(characters)},
            }
            self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

            super().__init__(
                bos_token=bos_token,
                eos_token=sep_token,
                sep_token=sep_token,
                cls_token=cls_token,
                pad_token=pad_token,
                mask_token=mask_token,
                unk_token=unk_token,
                add_prefix_space=False,
                model_max_length=model_max_length,
                padding_side=padding_side,
                **kwargs,
            )


        @property
        def vocab_size(self) -> int:
            return len(self._vocab_str_to_int)

        def _tokenize(self, text: str) -> List[str]:
            return list(text)

        def _convert_token_to_id(self, token: str) -> int:
            return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

        def _convert_id_to_token(self, index: int) -> str:
            return self._vocab_int_to_str[index]

        def convert_tokens_to_string(self, tokens):
            return "".join(tokens)

        def convert_token_vector_to_string(self, ivector):
            out_str = ""
            for i in ivector:
                out_str = out_str + self._convert_id_to_token(i.item())
            return out_str

        def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
            sep = [self.sep_token_id]
            cls = [self.cls_token_id]
            result = cls + token_ids_0 + sep
            if token_ids_1 is not None:
                result += token_ids_1 + sep
            return result

        def get_vocab(self):
            vocab = {chr(i): i for i in range(self.vocab_size)}
            vocab.update(self.added_tokens_encoder)
            return vocab

        def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False,
        ) -> List[int]:
            if already_has_special_tokens:
                return super().get_special_tokens_mask(
                    token_ids_0=token_ids_0,
                    token_ids_1=token_ids_1,
                    already_has_special_tokens=True,
                )

            result = [1] + ([0] * len(token_ids_0)) + [1]
            if token_ids_1 is not None:
                result += ([0] * len(token_ids_1)) + [1]
            return result

        def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
            sep = [self.sep_token_id]
            cls = [self.cls_token_id]

            result = len(cls + token_ids_0 + sep) * [0]
            if token_ids_1 is not None:
                result += len(token_ids_1 + sep) * [1]
            return result

        def get_config(self) -> Dict:
            return {
                "char_ords": [ord(ch) for ch in self.characters],
                "model_max_length": self.model_max_length,
            }

        @classmethod
        def from_config(cls, config: Dict) -> "CharacterTokenizer":
            cfg = {}
            cfg["characters"] = [chr(i) for i in config["char_ords"]]
            cfg["model_max_length"] = config["model_max_length"]
            return cls(**cfg)

        def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
            cfg_file = Path(save_directory) / "tokenizer_config.json"
            cfg = self.get_config()
            with open(cfg_file, "w") as f:
                json.dump(cfg, f, indent=4)

        @classmethod
        def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
            cfg_file = Path(save_directory) / "tokenizer_config.json"
            with open(cfg_file) as f:
                cfg = json.load(f)
            return cls.from_config(cfg)


    def create_genome_tokenizer():
        tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],
            model_max_length=seq_len,
            add_special_tokens=False,
            padding_side='left',
        )
        return tokenizer


    def complement(in_seq):
        out_seq = ""
        for idx, c in enumerate(in_seq):
            oc = "X"
            if c == 'A':
                oc = 'T'
            elif c == 'T':
                oc = 'A'
            elif c == 'C':
                oc = 'G'
            elif c == 'G':
                oc = 'C'
            elif c == 'N':
                oc = 'N'
            else:
                assert True == False, "char: {}".format(c)
            out_seq = out_seq + oc
        return out_seq
        

    class GenomeIterator:
        T2T_path = "dataset/ncbi_dataset/data/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"
        training_entries = ['NC_060925.1', 'NC_060926.1', 'NC_060927.1', 'NC_060928.1', 'NC_060929.1',
                            'NC_060931.1', 'NC_060932.1', 'NC_060933.1', 'NC_060934.1', 'NC_060935.1',
                            'NC_060936.1', 'NC_060937.1', 'NC_060938.1', 'NC_060939.1', 'NC_060941.1',
                            'NC_060942.1', 'NC_060943.1', 'NC_060944.1', 'NC_060945.1', 'NC_060946.1',
                            'NC_060947.1', 'NC_060948.1']
        validation_entries = ['NC_060930.1', 'NC_060940.1']

        def __init__(self, fasta_path, ds_entries, rnd_seed=0):
            assert Path(fasta_path).exists
            self.fasta = Fasta(fasta_path, one_based_attributes=False)

            dtype = np.dtype([('key', 'U20'), ('start', 'int_'), ('end', 'int_')])
            self.entry_ranges = np.empty(len(ds_entries), dtype=dtype)

            # only append entries of dataset
            count = 0
            for idx, k in enumerate(ds_entries):
                assert k in self.fasta.keys(), \
                    "FASTA file does not contain an entry with key {}".format(k)
                seq_len = len(self.fasta[k])
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

            left_bound = 0
            right_bound = len(self.fasta[key])

            seq = None
            if not(rev_compl):
                seq = self.fasta[key][:local_idx + 1][-self.seq_len:]
            else:
                seq = self.fasta[key][local_idx:][::-1][-self.seq_len:]
            assert seq != None

            # capitalize all nucleotides
            seq_str = str(seq).upper()

            # use complement when reverse
            if rev_compl:
                seq_str = complement(seq_str)
            # print(seq_str, len(seq_str))
            assert len(seq_str) <= self.seq_len

            tokens = self.tokenizer(seq_str, add_special_tokens=False, padding="max_length",
                                    max_length=self.seq_len, truncation=True)
            # TODO use CharTensor instead of LongTensor
            inpt = torch.LongTensor(tokens["input_ids"]).clone()

            # mask
            target = inpt.clone()
            inpt[-1] = self.tokenizer._vocab_str_to_int['[MASK]']
            return inpt, target


    class GenomeDataset(torch.utils.data.IterableDataset):
        def __init__(self, genomeIterator):
            super().__init__()
            self.genomeIterator = genomeIterator

        def config(self, tokenizer, seq_len):
            self.genomeIterator.config(tokenizer, seq_len)

        def __iter__(self):
            self.genomeIterator.reseed()
            return self.genomeIterator


    # code from https://github.com/apapiu/mamba_small_bench
    class MambaBlock(nn.Module):
        def __init__(self, embed_dim, dropout_level=0):
            super().__init__()

            self.mamba = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
            self.norm = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(dropout_level)

        def forward(self, x):
            # TODO check order of pytorch operations
            x = self.mamba(self.norm(x)) + x
            return self.dropout(x)


    class MambaTower(nn.Module):
        def __init__(self, embed_dim, n_layers, seq_len=None, dropout=0, global_pool=False):
            super().__init__()
            self.blocks = nn.Sequential(*[MambaBlock(embed_dim, dropout) for _ in range(n_layers)])
            self.global_pool = global_pool #for classification or other supervised learning.

        def forward(self, x):
            #for input (bs, n, d) it returns either (bs, n, d) or (bs, d) is global_pool
            out = self.blocks(x) if not self.global_pool else torch.mean(self.blocks(x),1)
            return out


    class MambaDNA(nn.Module):
        def __init__(self, vocab_size, embed_dim, seq_len, n_layers, dropout):
            super().__init__()

            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.tower = MambaTower(embed_dim, n_layers, seq_len=seq_len,
                                    dropout=dropout, global_pool=False)
            self.out_proj = nn.Sequential(nn.LayerNorm(embed_dim),
                                          nn.Linear(embed_dim, vocab_size))

        def forward(self, x):
            x = self.tower(self.embed(x))
            return self.out_proj(x)


    # assert torch.cuda.is_available() == True, "CUDA unavailable"
    # device = torch.device('cuda')
    # torch.cuda.set_device(device)
    # assert torch.cuda.utilization() == 0, "GPU in use!"
    # print("Using {} GPUs: {}".format(torch.cuda.device_count(), torch.cuda.get_device_name()))


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
        def __init__(self, mamba_model):
            super().__init__()
            self.mamba_model = mamba_model
            self.loss_fn = nn.CrossEntropyLoss()
            self.stime = None

        def forward(self, inpts):
            return self.mamba_model(inpts).logits

        def train_dataloader(self):
            seed = torch.distributed.get_rank()
            train_iter = GenomeIterator(GenomeIterator.T2T_path, GenomeIterator.training_entries, seed)
            train_ds = GenomeDataset(train_iter)

            tokenizer = create_genome_tokenizer()
            train_ds.config(tokenizer, seq_len)
            return DataLoader(train_ds, batch_size=batch_size_train)

        def training_step(self, batch, batch_idx):
            inpts, trgts = batch
            outpts = self(inpts)
            # print("inpts: {}, trgts: {}, outpts: {}".format(inpts.size(), trgts.size(), outpts.size()))
            loss = self.loss_fn(outpts.view(-1, outpts.size(-1)), trgts.view(-1))
            
            accuracy = comp_next_token_pred_acc(outpts, trgts)
            print("batch_idx {}: Loss {:.6f}; Masked prediction accuracy {:.4f}%".format(batch_idx, loss.item(), accuracy*100.0))
            self.log("train_loss", loss.item())
            self.log("train_accuracy", accuracy*100.0)
            return loss

        def val_dataloader(self):
            seed = torch.distributed.get_rank()
            val_iter = GenomeIterator(GenomeIterator.T2T_path, GenomeIterator.validation_entries, seed)
            val_ds = GenomeDataset(val_iter)

            tokenizer = create_genome_tokenizer()
            val_ds.config(tokenizer, seq_len)
            return DataLoader(val_ds, batch_size=batch_size_val)

        def validation_step(self, batch, batch_idx):
            inpts, trgts = batch
            outpts = self(inpts)
            
            accuracy = comp_next_token_pred_acc(outpts, trgts)
            print("batch_idx {} validataion: Masked prediction accuracy {:.4f}%".format(batch_idx, accuracy*100.0))
            self.log("val_accuracy", accuracy*100.0)

        def on_train_batch_start(self, batch, batch_idx):
            stime = time.time()
            if self.stime != None:
                print("Elapsed time: {:.2f}s".format(stime - self.stime))
            self.stime = stime
            
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                          weight_decay=weight_decay) #eps=epsilon, 
            return optimizer
        

    print("Not implemented: Model load/store")
    tokenizer = create_genome_tokenizer()

    mamba_config = MambaConfig(d_model=embed_dim, n_layer=n_layers, vocab_size=tokenizer.vocab_size,
                               ssm_cfg={}, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                               pad_vocab_size_multiple=8)
    model = MambaLMHeadModel(mamba_config)
    # dummy model
    #model = nn.Sequential(nn.Embedding(tokenizer.vocab_size, embed_dim), nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, tokenizer.vocab_size))

    # number of parameters of model
    print("#{} model parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    mambaDNA = LitMambaDNA(model)

    logger = TensorBoardLogger("tb_logs", name="mamba_model")
    trainer = L.Trainer(max_epochs=1, limit_train_batches=5, limit_val_batches=int(1), check_val_every_n_epoch=None, val_check_interval=5,
                        devices=8, accelerator="gpu", log_every_n_steps=1, logger=logger, strategy="ddp", use_distributed_sampler=False, profiler='simple')
    trainer.fit(mambaDNA)


if __name__ == '__main__':
    mamba_training()
