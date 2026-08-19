from transformers import AutoTokenizer

SMILES_TOKENIZER = AutoTokenizer.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)
