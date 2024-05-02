from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import datasets
import torch


def main():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    #tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf").to("cuda")

    input_texts = datasets.load_dataset("PY007/tokenized_proof_pile_test_neox", split="test")
    input_texts = input_texts.filter(lambda x: x["tokenized_len"] >= 32768, num_proc=64)
    input_texts = input_texts[:20]

    #print(input_texts)
    #print(tokenizer.bos_token)
    input_ids = tokenizer("Explain the difference between a FPGA and a CPU?", return_tensors="pt")["input_ids"].to("cuda")

    out = model.generate(input_ids, max_new_tokens=500).to('cpu')
    print(tokenizer.batch_decode(out))


if __name__ == '__main__':
    main()
