from transformers import AutoTokenizer  # noqa: I002

SMILES_TOKENIZER = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
SELFIES_TOKENIZER = AutoTokenizer.from_pretrained("HUBioDataLab/SELFormer")
GALACTICA_125M_TOKENIZER = AutoTokenizer.from_pretrained("facebook/galactica-125m")
