import argparse
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset, load_from_disk

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


def get(args):
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_identifier)
    hf_state_dict = hf_model.state_dict()
    hf_state_dict['backbone.embedding.weight'] = hf_state_dict.pop('backbone.embeddings.weight')
    hf_config = AutoConfig.from_pretrained(args.hf_identifier)

    os.makedirs(os.path.dirname(args.model_path + '/'), exist_ok=True)
    torch.save({'mamba_state_dict': hf_state_dict}, args.model_path + "/mamba_state_dict.pth")
    torch.save({'n_layer': hf_config.n_layer,
                'd_model': hf_config.hidden_size,
                'vocab_size': hf_config.vocab_size}, args.model_path + "/mamba_config.pth")

    hf_tokenizer = AutoTokenizer.from_pretrained(args.hf_identifier)
    hf_tokenizer.save_pretrained(args.model_path + "/tokenizer.pth")

    print("download slimpajama dataset with `git clone --jobs=<N> https://huggingface.co/datasets/cerebras/SlimPajama-627B` and set corresponding argument.")
    print("Note: `ulimit -n 8192`")
    assert os.environ.get("HF_HOME") is not None, \
             "HF_CACHE env variable not set; set to huggingface cache path"
    # TODO check if dataset exists, else print git clone command
    ds = load_dataset(args.slimpajama_path, num_proc=64) #, streaming=True,)


class SlimPajamaWrapper(IterableDataset):
    def __init__(self, sp_path, tokenizer, length, seed=42, split='train'):
        self.sp = load_dataset(sp_path, streaming=True, split=split).shuffle(seed, buffer_size=100000)
        self.sp_iter = iter(self.sp)
        self.tokenizer = tokenizer
        self.length = length
        self.rng = random.Random()
        self.rng.seed(seed)

    def __iter__(self):
        for e in self.sp_iter:
            txt = e['text']
            inpt_id = torch.tensor(self.tokenizer(txt)['input_ids'])
            # string is long enough
            if len(inpt_id) > self.length:
                # string too long; use only slice of it
                if len(inpt_id) > self.length + 1:
                    max_offset = len(inpt_id) - self.length - 1
                    offset = self.rng.randint(0, max_offset)
                    inpt_id = inpt_id[offset:offset+self.length+1]
                assert len(inpt_id) == self.length + 1, f"inpt_id list has incorrect length {len(inpt_id)}"
                inpt = inpt_id[:-1]
                assert len(inpt) == self.length, f"inpt list has incorrect length {len(inpt)}"
                trgt = inpt_id[1:]
                assert len(trgt) == self.length, f"trgt list has incorrect length {len(trgt)}"
                yield (inpt, trgt)
            # else:
                # print("tokenized string not long enough")


class SlimPajamaDataModule(L.LightningDataModule):
    def __init__(self, ds_path, tokenizer, length, batch_size_train, batch_size_val, batch_size_test, seed=42):
        super().__init__()
        self.ds_path = ds_path
        self.tokenizer = tokenizer
        self.length = length
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.seed = seed

    def setup(self, stage=None):
        self.ds_train = SlimPajamaWrapper(self.ds_path, self.tokenizer, self.length, seed=self.seed, split='train')
        self.ds_test = SlimPajamaWrapper(self.ds_path, self.tokenizer, self.length, seed=self.seed, split='test')
        self.ds_val = SlimPajamaWrapper(self.ds_path, self.tokenizer, self.length, seed=self.seed, split='validation')

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size_train)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size_val)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size_test)


class LitMamboros(L.LightningModule):
    def __init__(self, pretrained_mamboros, tokenizer, seq_len, lr, lr_scheduler_factor, weight_decay,
                 batch_size_train, batch_size_val):
        super().__init__()
        self.mamboros = pretrained_mamboros

        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss()

        self.seq_len = seq_len
        self.lr = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.weight_decay = weight_decay
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

        # self.save_hyperparameters(ignore=['mamborosDNA'])

    def forward(self, inpts):
        return self.mamboros(inpts).logits

    def training_step(self, batch, batch_idx):
        inpts, trgts = batch
        outpts = self(inpts)
        loss = self.loss_fn(outpts.view(-1, outpts.size(-1)), trgts.view(-1))
        self.log("train_loss", loss.item(), sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.mamboros.parameters(), lr=self.lr, betas=(0.9, 0.95),
                                      weight_decay=self.weight_decay) #eps=epsilon,
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=150,
                                                                  factor=self.lr_scheduler_factor, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}


def ftune(args):
    # training
    gpu_cnt = 1
    max_epochs = 10
    limit_train_batches = 16
    limit_val_batches = 32

    batch_size_train = 1
    batch_size_val = 1

    # optimizer
    lr = 8e-3
    lr_scheduler_factor = 0.85
    weight_decay = 0.1


    torch.set_float32_matmul_precision('medium')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path + "/tokenizer.pth")

    assert os.environ.get("HF_HOME") is not None, \
             "HF_CACHE env variable not set; set to huggingface cache path"
    sp_datamodule = SlimPajamaDataModule(args.slimpajama_path, tokenizer, 2000, batch_size_train, batch_size_val, 0)

    ssm_cfg = {'layer': 'Mamba1'}
    hf_config = torch.load(args.model_path + "/mamba_config.pth")
    mamba_config = MambaConfig(n_layer=hf_config['n_layer'], d_model=hf_config['d_model'], vocab_size=hf_config['vocab_size'],
                               ssm_cfg=ssm_cfg, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                               pad_vocab_size_multiple=1)

    pretrained_state_dict = torch.load(args.model_path + args.state_dict_in)['mamba_state_dict']
    pretrained_mamboros = MambaLMHeadModel(mamba_config)
    pretrained_mamboros.load_state_dict(pretrained_state_dict)
    pretrained_mamboros = pretrained_mamboros.to("cuda")

    seq_len = 5120
    l_mamboros = LitMamboros(pretrained_mamboros, tokenizer, seq_len, lr, lr_scheduler_factor,
                             weight_decay, batch_size_train, batch_size_val)


    logger = TensorBoardLogger("tb_logs", name="mamboros_model")
    ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(save_top_k=10, monitor="train_loss", save_on_train_epoch_end=True,
                                               verbose=True, every_n_epochs=10)

    trainer = L.Trainer(max_epochs=max_epochs, limit_train_batches=limit_train_batches,
                        limit_val_batches=limit_val_batches, check_val_every_n_epoch=5, gradient_clip_val=0.5,
                        gradient_clip_algorithm="norm", devices=gpu_cnt, accelerator="gpu",
                        precision='bf16-mixed', log_every_n_steps=1, logger=logger, strategy="ddp",
                        use_distributed_sampler=False, callbacks=[ckpt_cb])
    trainer.fit(l_mamboros, datamodule=sp_datamodule)

    torch.save({'mamba_state_dict': l_mamboros.mamboros.state_dict()}, args.model_path + args.state_dict_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="mamboros_ftuning")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    get_sp = subparsers.add_parser("get", help="get pretrained model from huggingface")
    get_sp.add_argument("--hf-identifier", default="state-spaces/mamba-2.8b-hf", help="huggingface identifier")
    get_sp.add_argument("--model-path", default="model_store/", help="path to store model and tokenizer")
    get_sp.set_defaults(func=get)

    ftune_sp = subparsers.add_parser("finetune", help="finetune model")
    ftune_sp.add_argument("--model-path", default="model_store/", help="path to load/store model")
    ftune_sp.add_argument("--state-dict-in", default="/mamba_state_dict.pth", help="input state dict file name")
    ftune_sp.add_argument("--state-dict-out", default="/mamba_state_dict.pth", help="output state dict file name")
    ftune_sp.add_argument("--slimpajama-path", default="/scratch/niklas/SlimPajama-627B", help="set path of slimpajama dataset")
    ftune_sp.set_defaults(func=ftune)

    args = parser.parse_args()
    args.func(args)
