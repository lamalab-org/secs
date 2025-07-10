from transformers import AutoTokenizer

SMILES_TOKENIZER = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

# POLYMER_NAME_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

# POLYMER_NAME_TOKENIZER = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')

POLYMER_NAME_TOKENIZER = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
