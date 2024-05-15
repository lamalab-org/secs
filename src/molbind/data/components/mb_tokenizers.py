from transformers import AutoTokenizer  # noqa: I002

SMILES_TOKENIZER = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
SELFIES_TOKENIZER = AutoTokenizer.from_pretrained("HUBioDataLab/SELFormer")
LLAMA_3_8B_TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/llama-3-7b-hf")
