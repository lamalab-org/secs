from transformers import AutoTokenizer

SMILES_TOKENIZER = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
SELFIES_TOKENIZER = AutoTokenizer.from_pretrained("HUBioDataLab/SELFormer")
