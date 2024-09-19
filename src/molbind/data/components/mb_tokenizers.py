from transformers import AutoTokenizer

SMILES_TOKENIZER = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
SELFIES_TOKENIZER = AutoTokenizer.from_pretrained("HUBioDataLab/SELFormer")
DESCRIPTION_TOKENIZER = None
