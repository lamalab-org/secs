from transformers import AutoTokenizer  # noqa: I002

SMILES_TOKENIZER = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
SELFIES_TOKENIZER = AutoTokenizer.from_pretrained("HUBioDataLab/SELFormer")
DESCRIPTION_TOKENIZER = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
