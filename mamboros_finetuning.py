import argparse
import os
import random
import re

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, IterableDataset

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset, load_from_disk

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
# from mamba_ssm.modules.block import Block


def get(args):
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_identifier)
    hf_state_dict = hf_model.to("cpu").state_dict()
    hf_state_dict['backbone.embedding.weight'] = hf_state_dict.pop('backbone.embeddings.weight')
    hf_config = AutoConfig.from_pretrained(args.hf_identifier)

    os.makedirs(os.path.dirname(args.model_path + '/'), exist_ok=True)
    torch.save({'mamba_state_dict': hf_state_dict}, args.model_path + "/mamba_state_dict.pth")
    torch.save({'n_layer': hf_config.n_layer,
                'd_model': hf_config.hidden_size,
                'vocab_size': hf_config.vocab_size}, args.model_path + "/mamba_config.pth")

    hf_tokenizer = AutoTokenizer.from_pretrained(args.hf_identifier)
    hf_tokenizer.save_pretrained(args.model_path + "/tokenizer.pth")

    if args.prep_dataset is True:
        print("download slimpajama dataset with `git clone --jobs=<N> https://huggingface.co/datasets/cerebras/SlimPajama-627B` and set corresponding argument.")
        print("Note: `ulimit -n 8192`")
        assert os.environ.get("HF_HOME") is not None, \
                 "HF_CACHE env variable not set; set to huggingface cache path"
        # TODO check if dataset exists, else print git clone command
        ds = load_dataset(args.slimpajama_path, num_proc=64) #, streaming=True,)

    # cache dataset for perplexity measurement
    ppls_ds = load_dataset("PY007/tokenized_proof_pile_test_neox", split="test")


class SlimPajamaWrapper(IterableDataset):
    def __init__(self, sp_path, tokenizer, pseudo_length, length, batch_size, seed=42, split='train'):
        self.sp = load_dataset(sp_path, streaming=True, split=split).shuffle(seed, buffer_size=100000)
        self.sp_iter = iter(self.sp)
        self.tokenizer = tokenizer
        self.pseudo_length = pseudo_length
        self.length = length
        assert self.pseudo_length % self.length == 0, f"expect pseudo_length ({self.pseudo_length}) to be multiple of length ({self.length})"
        assert self.pseudo_length >= self.length, "expect pseudo_length to be greater or equal to length"
        self.length_ratio = self.pseudo_length // self.length
        self.batch_size = batch_size
        self.rng = random.Random()
        self.rng.seed(seed)
        self.buffer = [None] * self.batch_size
        self.sub_iter_idx = 0

    def __iter__(self):
        while True:
            if self.sub_iter_idx >= (self.batch_size * self.length_ratio):
                self.sub_iter_idx = 0
                self.buffer = [None] * self.batch_size

            # fill buffer if empty
            for i in range(len(self.buffer)):
                while self.buffer[i] is None:
                    try:
                        txt = next(self.sp_iter)['text']
                    except StopIteration:
                        return
                    inpt_id = torch.tensor(self.tokenizer(txt)['input_ids'])
                    # string is long enough
                    if len(inpt_id) > self.pseudo_length:
                        # string too long; use only slice of it
                        if len(inpt_id) > self.pseudo_length + 1:
                            max_offset = len(inpt_id) - self.pseudo_length - 1
                            offset = self.rng.randint(0, max_offset)
                            inpt_id = inpt_id[offset:offset+self.pseudo_length+1]
                        assert len(inpt_id) == self.pseudo_length + 1, f"inpt_id list has incorrect length {len(inpt_id)}"
                        self.buffer[i] = inpt_id

            buffer_idx = self.sub_iter_idx % self.batch_size
            buffer_element = self.sub_iter_idx // self.batch_size

            range_start = buffer_element * self.length
            range_end = range_start + self.length
            inpt = self.buffer[buffer_idx][range_start:range_end].clone()
            trgt = self.buffer[buffer_idx][range_start+1:range_end+1].clone()
            assert len(inpt) == self.length, f"inpt has incorrect length {len(inpt)}"
            assert len(trgt) == self.length, f"trgt has incorrect length {len(trgt)}"

            self.sub_iter_idx += 1
            yield (inpt, trgt)


class SlimPajamaDataModule(L.LightningDataModule):
    def __init__(self, ds_path, tokenizer, pseudo_length, length, batch_size_train, batch_size_val, batch_size_test, seed=42):
        super().__init__()
        self.ds_path = ds_path
        self.tokenizer = tokenizer
        self.pseudo_length = pseudo_length
        self.length = length
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.seed = seed

    def setup(self, stage=None):
        if dist.is_initialized():
            self.seed = self.seed + dist.get_rank()
        self.ds_train = SlimPajamaWrapper(self.ds_path, self.tokenizer, self.pseudo_length, self.length, batch_size=self.batch_size_train, seed=self.seed, split='train')
        # self.ds_train = SlimPajamaWrapper(self.ds_path, self.tokenizer, self.pseudo_length, self.length, batch_size=self.batch_size_train, seed=self.seed, split='test')
        # self.ds_test = SlimPajamaWrapper(self.ds_path, self.tokenizer, self.pseudo_length, self.length, batch_size=self.batch_size_test, seed=self.seed, split='test')
        # self.ds_val = SlimPajamaWrapper(self.ds_path, self.tokenizer, self.pseudo_length, self.length, batch_size=self.batch_size_val, seed=self.seed, split='validation')

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size_train)

    # def val_dataloader(self):
    #     return DataLoader(self.ds_val, batch_size=self.batch_size_val)

    # def test_dataloader(self):
    #     return DataLoader(self.ds_test, batch_size=self.batch_size_test)

    def testbench(args):
        length = 1024
        tokenizer = AutoTokenizer.from_pretrained(args.model_path + "/tokenizer.pth")
        batch_size_train = 5
        batch_size_val = 1

        # test pseudo_length equal to length
        pseudo_length = length

        length_ratio = pseudo_length // length
        sp_datamodule = SlimPajamaDataModule(args.slimpajama_path, tokenizer, pseudo_length, length, batch_size_train, batch_size_val, 42)
        sp_datamodule.setup()
        dl = sp_datamodule.train_dataloader()

        for i, (mamboros_batch_inpt, mamboros_batch_trgt) in enumerate(dl):
            for i2 in range(batch_size_train):
                ref = dl.dataset.buffer[i2]

                assert torch.equal(mamboros_batch_inpt[i2,:], ref[:-1])
                assert torch.equal(mamboros_batch_trgt[i2,:], ref[1:])
            print(f"1.{i} inpt and trgt match with reference")

            if i >= 10:
                break

        print("Mamboros dataloader working as expected for pseudo_length==length!")

        # test pseudo_length multiple of length
        pseudo_length = 3 * length

        length_ratio = pseudo_length // length
        sp_datamodule = SlimPajamaDataModule(args.slimpajama_path, tokenizer, pseudo_length, length, batch_size_train, batch_size_val, 42)
        sp_datamodule.setup()
        dl = sp_datamodule.train_dataloader()

        mamboros_batch_inpt = None
        mamboros_batch_trgt = None
        for i, (inpt, trgt) in enumerate(dl):
            if mamboros_batch_inpt is None:
                mamboros_batch_inpt = inpt
                mamboros_batch_trgt = trgt
            else:
                mamboros_batch_inpt = torch.cat((mamboros_batch_inpt, inpt))
                mamboros_batch_trgt = torch.cat((mamboros_batch_trgt, trgt))

            print(f"{i} {inpt.shape} {mamboros_batch_inpt.shape}")
            if i % length_ratio == length_ratio - 1:
                mamboros_batch_inpt = mamboros_batch_inpt.view([-1, batch_size_train, length])
                mamboros_batch_inpt = torch.transpose(mamboros_batch_inpt, 0, 1).reshape([-1, pseudo_length])

                mamboros_batch_trgt = mamboros_batch_trgt.view([-1, batch_size_train, length])
                mamboros_batch_trgt = torch.transpose(mamboros_batch_trgt, 0, 1).reshape([-1, pseudo_length])

                for i2 in range(batch_size_train):
                    ref = dl.dataset.buffer[i2]

                    assert torch.equal(mamboros_batch_inpt[i2,:], ref[:-1])
                    assert torch.equal(mamboros_batch_trgt[i2,:], ref[1:])
                print(f"2.{i//length_ratio} inpt and trgt match with reference")

                mamboros_batch_inpt = None
                mamboros_batch_trgt = None


            if i >= 10 * length_ratio + 1:
                break

        print("Mamboros dataloader working as expected for pseudo_length>length!")


class LitMamboros(L.LightningModule):
    def __init__(self, pretrained_mamboros, tokenizer, lr, lr_scheduler_factor, weight_decay,
                 batch_size_train, batch_size_val):
        super().__init__()
        self.mamboros = pretrained_mamboros

        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss()

        self.lr = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.weight_decay = weight_decay
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

        # self.save_hyperparameters(ignore=['mamborosDNA'])

    def forward(self, inpts):
        return self.mamboros(inpts).logits

    def predict_step(self, batch, batch_idx):
        match batch:
            case (inpt, trgt):
                preds = self(inpt)
                cross_entropy = torch.nn.functional.cross_entropy(preds.view(-1, preds.size(-1)), trgt.view(-1), reduction='none')
                return cross_entropy.view(trgt.shape)
            case inpt:
                return self(inpt)

    def training_step(self, batch, batch_idx):
        inpts, trgts = batch
        outpts = self(inpts)
        loss = self.loss_fn(outpts.view(-1, outpts.size(-1)), trgts.view(-1))
        self.log("train_loss", loss.item(), sync_dist=True)

        return loss

    # def on_train_batch_start(self, batch, batch_idx):
    #     raise NotImplementedError()

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     raise NotImplementedError()

    # def on_fit_start(self):


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.mamboros.parameters(), lr=self.lr, betas=(0.9, 0.95),
                                      weight_decay=self.weight_decay) #eps=epsilon,
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=150,
                                                                  factor=self.lr_scheduler_factor, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}


def ftune(args):
    if args.check_ds_dl:
        SlimPajamaDataModule.testbench(args)

    torch.cuda.memory._record_memory_history(max_entries=500000)

    # training
    gpu_cnt = 6
    max_epochs = 10
    limit_train_batches = 4 * 10 #25 #50 #* 20
    limit_val_batches = 4 * 100

    batch_size_train = 4
    batch_size_val = 4

    # optimizer
    lr = 5e-5
    lr_scheduler_factor = 0.85
    weight_decay = 0.1


    torch.set_float32_matmul_precision('medium')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path + "/tokenizer.pth")

    assert os.environ.get("HF_HOME") is not None, \
             "HF_CACHE env variable not set; set to huggingface cache path"
    length = args.context_length
    pseudo_length = args.pseudo_context_length
    length_ratio = pseudo_length // length
    assert limit_train_batches % length_ratio == 0, f"limit_train_batches ({limit_train_batches}) expected to be multiple of length_ratio ({length_ratio})"
    assert limit_val_batches % length_ratio == 0, f"limit_val_batches ({limit_val_batches}) expected to be multiple of length_ratio ({length_ratio})"
    sp_datamodule = SlimPajamaDataModule(args.slimpajama_path, tokenizer, pseudo_length, length, batch_size_train, batch_size_val, 42)

    ssm_cfg = {'max_hstate_trnsf_cnt': length_ratio-1}
    hf_config = torch.load(args.model_path + "/mamba_config.pth")
    mamba_config = MambaConfig(n_layer=hf_config['n_layer'], d_model=hf_config['d_model'], vocab_size=hf_config['vocab_size'],
                               ssm_cfg=ssm_cfg, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                               pad_vocab_size_multiple=1)

    pretrained_state_dict = torch.load(args.model_path + args.state_dict_in)['mamba_state_dict']
    # check state_dict is on CPU
    for (key, value) in pretrained_state_dict.items():
        assert type(key) is not torch.Tensor
        assert type(value) is torch.Tensor
        assert value.device == torch.device("cpu"), f"expected state_dict of pretrained model to be on cpu; instead is {value.device}"

    pretrained_mamboros = MambaLMHeadModel(mamba_config)
    pretrained_mamboros.load_state_dict(pretrained_state_dict)

    l_mamboros = LitMamboros(pretrained_mamboros, tokenizer, lr, lr_scheduler_factor,
                             weight_decay, batch_size_train, batch_size_val)


    logger = TensorBoardLogger("tb_logs", name="mamboros_model")
    ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(save_top_k=10, monitor="train_loss", save_on_train_epoch_end=True,
                                               verbose=True, every_n_epochs=10)

    # policy = {Block, }
    # strategy = FSDPStrategy(sharding_strategy="SHARD_GRAD_OP", activation_checkpointing_policy=policy, auto_wrap_policy=policy)
    # strategy = FSDPStrategy(sharding_strategy="FULL_SHARD", activation_checkpointing_policy=policy, auto_wrap_policy=policy)

    trainer = L.Trainer(max_epochs=max_epochs, limit_train_batches=limit_train_batches,
                        limit_val_batches=limit_val_batches, check_val_every_n_epoch=5, #gradient_clip_val=0.5, gradient_clip_algorithm="norm",
                        devices=gpu_cnt, accelerator="gpu",
                        precision='bf16-mixed', log_every_n_steps=1, logger=logger, strategy="fsdp",
                        use_distributed_sampler=False, callbacks=[ckpt_cb])
    trainer.fit(l_mamboros, datamodule=sp_datamodule)

    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    # store model parameters
    if args.state_dict_out is not None:
        if trainer.is_global_zero:
            print("storing the model parameters")
        tmp_ckpt_file = 'tmp_ckpt'
        trainer.save_checkpoint(tmp_ckpt_file)
        if trainer.is_global_zero:
            tmp_ckpt = torch.load(tmp_ckpt_file)
            state_dict = tmp_ckpt['state_dict']
            state_dict = {re.search(r'^[^.]*\.(.*)', key).group(1): value for key, value in state_dict.items()}
            for (key, value) in state_dict.items():
                assert type(key) is not torch.Tensor
                assert type(value) is torch.Tensor
                assert value.device == torch.device("cpu"), f"expected state_dict of pretrained model to be on cpu; instead is {value.device}"
            torch.save({'mamba_state_dict': state_dict}, args.model_path + args.state_dict_out)
            os.remove(tmp_ckpt_file)
            print("done")


class PPLAnalysesDS(Dataset):
    def __init__(self, context_length=None, pseudo_context_length=None, batch_size=None, size=100):
        ppls_ds = load_dataset("PY007/tokenized_proof_pile_test_neox", split="test")
        ppls_ds = ppls_ds.filter(lambda x: x["tokenized_len"] >= 32768, num_proc=64)
        assert len(ppls_ds) >= size, "not enough elements in ppls_ds"
        ppls_ds = ppls_ds[:size]
        self.encoded_texts = ppls_ds["input_ids"]
        assert len(self.encoded_texts) == size, "expected {size} number of encoded_texts; got {self.encoded_texts.size(0)}"

        # for i, t in enumerate(self.encoded_texts):
        #     print(f"{i}: {t[:20]}")
        #     if i > 5:
        #         break

        self.context_length = context_length
        self.pseudo_context_length = pseudo_context_length
        self.batch_size = batch_size
        self.size = size

    def context_length_ratio(self):
        return self.pseudo_context_length // self.context_length

    def config(self, context_length, pseudo_context_length, batch_size):
        self.context_length = context_length
        self.pseudo_context_length = pseudo_context_length
        assert self.pseudo_context_length % self.context_length == 0, f"expect pseudo_context_length ({self.pseudo_context_length}) to be multiple of context_length ({self.context_length})"
        assert self.pseudo_context_length >= self.context_length, "expect pseudo_context_length to be at least context_length"
        self.batch_size = batch_size
        assert self.size % self.batch_size == 0, "expect batch_size to be multiple of size"

    def __len__(self):
        return self.size * self.context_length_ratio()

    def __getitem__(self, idx):
        mamboros_batch_idx = idx // (self.batch_size * self.context_length_ratio())
        local_idx = idx % (self.batch_size * self.context_length_ratio())

        seq_idx = mamboros_batch_idx * self.batch_size + local_idx % self.batch_size
        seq_element_idx = local_idx // self.batch_size

        start_range = seq_element_idx * self.context_length
        end_range = start_range + self.context_length

        inpt = self.encoded_texts[seq_idx][start_range:end_range]
        trgt = self.encoded_texts[seq_idx][start_range+1:end_range+1]
        # print(f"DEBUG: seq_idx {seq_idx}")
        # print(f"DEBUG: seq_element_idx {seq_element_idx}")
        # print(f"DEBUG: start_range {start_range}")
        # print(f"DEBUG: end_range {end_range}")
        return (torch.tensor(inpt), torch.tensor(trgt))

    def test_dataset():
        context_length = 1024
        pseudo_context_length = context_length
        batch_size = 4

        ppl_ds = PPLAnalysesDS(size=12)

        # test context_length equal to pseudo_context_length
        ppl_ds.config(context_length, pseudo_context_length, batch_size)

        test_vectors0 = [
                # item-idx, seq-idx, start-range, end_range
                [0, 0, 0, 1024],
                [1, 1, 0, 1024],
                [2, 2, 0, 1024],
                [3, 3, 0, 1024],
                [4, 4, 0, 1024],
                [5, 5, 0, 1024],
                [6, 6, 0, 1024],
                [7, 7, 0, 1024],
                [8, 8, 0, 1024],
                [9, 9, 0, 1024],
                [10, 10, 0, 1024],
                [11, 11, 0, 1024],
                # [12, 12, 0, 1024],
            ]

        for i, (item_idx, seq_idx, seq_start_range, seq_end_range) in enumerate(test_vectors0):
            print(f"test 1.{i} (item_idx {item_idx})")
            # print(f"seq_idx {seq_idx}")
            # print(f"seq_start_range {seq_start_range}")
            # print(f"seq_end_range {seq_end_range}")

            (inpt, trgt) = ppl_ds.__getitem__(item_idx)
            inpt2 = torch.tensor(ppl_ds.encoded_texts[seq_idx][seq_start_range:seq_end_range])
            trgt2 = torch.tensor(ppl_ds.encoded_texts[seq_idx][seq_start_range+1:seq_end_range+1])

            # print("inpt")
            # print(inpt[:10], inpt[-10:])
            # print(inpt2[:10], inpt2[-10:])

            # print("trgt")
            # print(trgt[:10], trgt[-10:])
            # print(trgt2[:10], trgt2[-10:])

            # print(len(inpt))
            # print(len(trgt))
            # print("")

            assert torch.equal(inpt, inpt2)
            assert torch.equal(trgt, trgt2)
            assert len(inpt) == context_length
            assert len(trgt) == context_length

        pseudo_context_length = 3 * context_length
        ppl_ds.config(context_length, pseudo_context_length, batch_size)

        test_vectors2 = [
                # item-idx, seq-idx, start-range, end_range
                [0, 0, 0, 1024],
                [1, 1, 0, 1024],
                [2, 2, 0, 1024],
                [3, 3, 0, 1024],
                [4, 0, 1024, 2048],
                [5, 1, 1024, 2048],
                [6, 2, 1024, 2048],
                [7, 3, 1024, 2048],
                [8, 0, 2048, 3072],
                [9, 1, 2048, 3072],
                [10, 2, 2048, 3072],
                [11, 3, 2048, 3072],
                [12, 4, 0, 1024],
                [13, 5, 0, 1024],
                [14, 6, 0, 1024],
                [15, 7, 0, 1024],
                [16, 4, 1024, 2048],
                [17, 5, 1024, 2048],
                [18, 6, 1024, 2048],
                [19, 7, 1024, 2048],
                [20, 4, 2048, 3072],
                [21, 5, 2048, 3072],
                [22, 6, 2048, 3072],
                [23, 7, 2048, 3072],
                [24, 8, 0, 1024],
            ]

        for i, (item_idx, seq_idx, seq_start_range, seq_end_range) in enumerate(test_vectors2):
            print(f"test 2.{i} (item_idx {item_idx})")
            # print(f"seq_idx {seq_idx}")
            # print(f"seq_start_range {seq_start_range}")
            # print(f"seq_end_range {seq_end_range}")

            (inpt, trgt) = ppl_ds.__getitem__(item_idx)
            inpt2 = torch.tensor(ppl_ds.encoded_texts[seq_idx][seq_start_range:seq_end_range])
            trgt2 = torch.tensor(ppl_ds.encoded_texts[seq_idx][seq_start_range+1:seq_end_range+1])

            # print("inpt")
            # print(inpt[:10], inpt[-10:])
            # print(inpt2[:10], inpt2[-10:])

            # print("trgt")
            # print(trgt[:10], trgt[-10:])
            # print(trgt2[:10], trgt2[-10:])

            # print(len(inpt))
            # print(len(trgt))
            # print("")

            assert torch.equal(inpt, inpt2)
            assert torch.equal(trgt, trgt2)
            assert len(inpt) == context_length
            assert len(trgt) == context_length


        test_vectors3 = [
                # [item_idxs], seq_idx, start_range, end_range
                [[0, 4, 8], 0, 0, 3072],
                [[1, 5, 9], 1, 0, 3072],
                [[2, 6, 10], 2, 0, 3072],
                [[3, 7, 11], 3, 0, 3072],
                [[12, 16, 20], 4, 0, 3072],
            ]
        for i, (item_idxs, seq_idx, start_range, end_range) in enumerate(test_vectors3):
            print(f"test 3.{i}")
            inpt = torch.cat(tuple(ppl_ds.__getitem__(item_idx)[0] for item_idx in item_idxs))
            # inpt = []
            # for item_idx in item_idxs:
            #     inpt = inpt + ppl_ds.__getitem__(item_idx)[0]

            assert torch.equal(inpt, torch.tensor(ppl_ds.encoded_texts[seq_idx][start_range:end_range]))
            assert len(inpt) == end_range - start_range

        print("PPLAnalysisDS dataset test successful!")

    def test_dataloader():
        context_length = 1024
        pseudo_context_length = 3 * context_length
        batch_size = 4

        ppl_ds = PPLAnalysesDS(size=12)
        ppl_ds.config(context_length, pseudo_context_length, batch_size)

        dl = DataLoader(ppl_ds, batch_size=batch_size, shuffle=False)

        for i, (inpts, trgts) in enumerate(dl):
            print(f"test dataloader batch {i}")

            idxs = list(range(i * batch_size, i * batch_size + batch_size))
            inpts2 = torch.stack(tuple(ppl_ds.__getitem__(idx)[0] for idx in idxs), dim=0)
            trgts2 = torch.stack(tuple(ppl_ds.__getitem__(idx)[1] for idx in idxs), dim=0)

            # print(inpts)
            # print(inpts2)

            assert torch.equal(inpts, inpts2)
            assert torch.equal(trgts, trgts2)
            assert inpts.shape == inpts2.shape
            assert trgts.shape == trgts2.shape

        print("PPLAnalysisDS dataloader test successful!")


def ppl_analysis(args):
    context_length = 1024
    pseudo_context_length = 20 * context_length
    context_length_ratio = pseudo_context_length // context_length
    batch_size = 5
    test_batch_count = 100

    # test perplexity analysis dataset/dataloader
    if args.check_ds_dl:
        PPLAnalysesDS.test_dataset()
        PPLAnalysesDS.test_dataloader()
    ppl_ds = PPLAnalysesDS(size=batch_size*test_batch_count)

    torch.set_float32_matmul_precision('medium')

    hf_config = torch.load(args.model_path + "/mamba_config.pth")
    pretrained_state_dict = torch.load(args.model_path + args.state_dict)['mamba_state_dict']

    ppl_ds.config(context_length, pseudo_context_length, batch_size)
    dl = DataLoader(ppl_ds, batch_size=batch_size)

    hstate_trnsf_cnt = context_length_ratio - 1
    ssm_cfg = {'max_hstate_trnsf_cnt': hstate_trnsf_cnt}
    mamba_config = MambaConfig(n_layer=hf_config['n_layer'], d_model=hf_config['d_model'], vocab_size=hf_config['vocab_size'],
                               ssm_cfg=ssm_cfg, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                               pad_vocab_size_multiple=1)
    mamboros = MambaLMHeadModel(mamba_config)
    mamboros.load_state_dict(pretrained_state_dict)
    l_mamboros = LitMamboros(mamboros, None, None, None, None, None, None)

    trainer = L.Trainer(max_epochs=1, devices=1, accelerator="gpu",
                        # limit_predict_batches=context_length_ratio*5,
                        precision='bf16-mixed')

    cross_entropy = trainer.predict(l_mamboros, dl)

    cross_entropy = torch.stack(cross_entropy, dim=0)
    cross_entropy = cross_entropy.view([-1, context_length_ratio] + list(cross_entropy.shape)[1:])
    cross_entropy = torch.transpose(cross_entropy, 1, 2)
    cross_entropy = cross_entropy.reshape([cross_entropy.size(0) * cross_entropy.size(1), cross_entropy.size(2) * cross_entropy.size(3)])

    ppl = torch.exp(cross_entropy)

    ppl_np = ppl.numpy()
    np.save(args.file, ppl_np)
    print(f"Perplexity analysis completed (data saved to the file {args.file})")


def ppl_vis_gen(args):
    ppl_np = np.load(args.file)

    x_axis = np.arange(1, ppl_np.shape[1]+1)
    y_axis = np.median(ppl_np, axis=0)
    # print(y_axis.shape)
    # print(y_axis[100:130])

    plt.scatter(x_axis, y_axis, marker='.', s=6)
    plt.grid(True)
    plt.xlabel("Context Length")
    plt.ylabel("Log of Median Perplexity")
    plt.yscale("log")

    plt.savefig(args.fig_file, dpi=500, bbox_inches='tight')
    print(f"plot {args.fig_file} created")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="mamboros_ftuning")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    get_sp = subparsers.add_parser("get", help="get pretrained model from huggingface")
    get_sp.add_argument("--hf-identifier", default="state-spaces/mamba-2.8b-hf", help="huggingface identifier")
    get_sp.add_argument("--model-path", default="model_store/", help="path to store model and tokenizer")
    get_sp.add_argument("--prep-dataset", action="store_true", help="prepare/decompress dataset")
    get_sp.set_defaults(func=get)

    ftune_sp = subparsers.add_parser("finetune", help="finetune model")
    ftune_sp.add_argument("--model-path", default="model_store/", help="path to load/store model")
    ftune_sp.add_argument("--state-dict-in", default="/mamba_state_dict.pth", help="input state dict file name")
    ftune_sp.add_argument("--state-dict-out", default=None, help="output state dict file name")
    ftune_sp.add_argument("--slimpajama-path", default="/scratch/niklas/SlimPajama-627B", help="set path of slimpajama dataset")
    ftune_sp.add_argument("--check-ds-dl", action="store_true", help="check mamboros dataset/dataloader")
    ftune_sp.add_argument("--context-length", default=1024, type=int, help="set context-length")
    ftune_sp.add_argument("--pseudo-context-length", default=4096, type=int, help="set pseudo-context-length")
    ftune_sp.set_defaults(func=ftune)

    ppl_sp = subparsers.add_parser("compute-ppl", help="compute perplexity over context length")
    ppl_sp.add_argument("--model-path", default="model_store/", help="path to load/store model")
    ppl_sp.add_argument("--state-dict", default="/mamba_state_dict.pth", help="state dict file name")
    ppl_sp.add_argument("--check-ds-dl", action="store_true", help="check dataset/dataloader of perplexity analysis")
    ppl_sp.add_argument("--file", default="ppl_analysis.npy", help="numpy output file path")
    ppl_sp.set_defaults(func=ppl_analysis)

    ppl_vis_sp = subparsers.add_parser("generate-ppl-vis", help="generate visualization of perplexity analysis")
    ppl_vis_sp.add_argument("--file", default="ppl_analysis.npy", help="numpy input file path")
    ppl_vis_sp.add_argument("--fig-file", default="fig.png", help="figure output file path")
    ppl_vis_sp.set_defaults(func=ppl_vis_gen)

    args = parser.parse_args()
    args.func(args)
