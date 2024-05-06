from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import torch

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    hf_model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf") #.to("cuda")
    hf_state_dict = hf_model.state_dict()
    hf_state_dict['backbone.embedding.weight'] = hf_state_dict.pop('backbone.embeddings.weight')

    mamba_config = MambaConfig(n_layer=64, d_model=2560, vocab_size=50280,
                               ssm_cfg={}, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                               pad_vocab_size_multiple=1)
    model = MambaLMHeadModel(mamba_config)
    model.load_state_dict(hf_state_dict)
    model = model.to("cuda")
    
    input_texts = datasets.load_dataset("PY007/tokenized_proof_pile_test_neox", split="test")
    input_texts = input_texts.filter(lambda x: x["tokenized_len"] >= 32768, num_proc=64)
    input_texts = input_texts[:20]

    tokens = [x for x in range(2048, 122889, 2048)]

    input_ids = tokenizer("Explain the difference between a FPGA and a CPU?", return_tensors="pt")["input_ids"].to("cuda")

    out = model.generate(input_ids, max_length=500).to("cpu")
    print("mamba")
    for a in tokenizer.batch_decode(out):
        print(a)

    hf_out = hf_model.to("cuda").generate(input_ids, max_new_tokens=500).to("cpu")
    print("hf_mamba")
    for a in tokenizer.batch_decode(hf_out):
        print(a)


if __name__ == '__main__':
    main()
