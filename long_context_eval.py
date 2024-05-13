import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import torch

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


def compute(args):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    hf_model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf")
    hf_state_dict = hf_model.state_dict()
    hf_state_dict['backbone.embedding.weight'] = hf_state_dict.pop('backbone.embeddings.weight')

    input_texts = datasets.load_dataset("PY007/tokenized_proof_pile_test_neox", split="test")
    input_texts = input_texts.filter(lambda x: x["tokenized_len"] >= 32768, num_proc=64)
    input_texts = input_texts[:50]

    df = None

    for context_length in range(1024, 15*1024+1, 1024):
        torch.cuda.empty_cache()
        ppls = []
        encoded_texts = input_texts["input_ids"]

        encoded_texts = [x[0:context_length] for x in encoded_texts]

        for idx in range(0, len(encoded_texts)):
            ssm_cfg = {}
            if args.context_length is not None:
                hstate_trnsf_cnt = (context_length // args.context_length) - 1
                ssm_cfg = {'max_hstate_trnsf_cnt': hstate_trnsf_cnt}

            mamba_config = MambaConfig(n_layer=64, d_model=2560, vocab_size=50280,
                                       ssm_cfg=ssm_cfg, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                                       pad_vocab_size_multiple=1)

            model = MambaLMHeadModel(mamba_config)
            model.load_state_dict(hf_state_dict)
            model = model.to("cuda")
            model.eval()

            input_ids = torch.tensor([encoded_texts[idx]]).to("cuda")
            target_ids = input_ids[:, 1:].clone().contiguous()

            with torch.no_grad():
                if args.context_length is None:
                    logits = model(input_ids).logits[:, :-1, :].contiguous()
                else:
                    logits = torch.zeros(target_ids.size(0), target_ids.size(1), mamba_config.vocab_size).to("cuda")
                    for i in range(0, context_length-args.context_length-1, args.context_length):
                        logits[0, i:i+args.context_length, :] = \
                               model(input_ids[:, i:i+args.context_length]).logits[:, :, :].contiguous()
                    logits[0, -args.context_length+1:, :] = model(input_ids[:, -args.context_length:]).logits[:,:-1,:].contiguous()

                cross_entropy = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction='mean')
                ppl = torch.exp(cross_entropy)
                ppls.append(ppl.item())

        df_new = pd.DataFrame([{'context_length': context_length, 'ppls': ppls}])
        df = df_new if df is None else pd.concat([df, df_new], ignore_index=True)
        print(f"context length {context_length} done!")

    df.to_pickle(args.file)


def visualize(args):
    fig = plt.figure()
    for file in args.files:
        df = pd.read_pickle(file)

        vis_data = np.zeros((len(df['context_length']), 3))
        for (idx, context_length) in enumerate(df['context_length']):
            ppls = np.array(df[df['context_length']==context_length]['ppls'].item())

            vis_data[idx, 0] = context_length
            vis_data[idx, 1] = ppls.mean()
            vis_data[idx, 2] = ppls.std()

        label = Path(file).stem
        plt.errorbar(vis_data[:, 0], vis_data[:, 1], yerr=vis_data[:, 2], marker='x', capsize=6, label=label)

    plt.xlabel("Context Length")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.savefig(args.fig_file, dpi=300, bbox_inches='tight')
    print("plot created")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="long_context_eval")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    comp_sp = subparsers.add_parser("compute", help="compute perplexity for multiple context lengths")
    comp_sp.add_argument("--file", type=str, default="data.pkl", help="output file path", metavar="path/to/data.pkl")
    comp_sp.add_argument("--context_length", type=int, default=None, help="native mamboros context length")
    comp_sp.set_defaults(func=compute)

    vis_sp = subparsers.add_parser("visualize", help="visualize perplexity")
    vis_sp.add_argument("--files", type=str, default=["data.pkl"], nargs='+', help="input file path", metavar="path/to/data.pkl")
    vis_sp.add_argument("--fig_file", type=str, default="fig.png", help="output figure path", metavar="path/to/fig.png")
    vis_sp.set_defaults(func=visualize)

    args = parser.parse_args()
    args.func(args)
