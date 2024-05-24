from transformers import AutoTokenizer, RobertaTokenizerFast  # noqa: I002

SMILES_TOKENIZER = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
SELFIES_TOKENIZER = AutoTokenizer.from_pretrained("HUBioDataLab/SELFormer")
TEXT_TOKENIZER = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
