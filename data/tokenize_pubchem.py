import multiprocessing
import os

import polars as pl
from loguru import logger

from molbind.data.components.mb_tokenizers import SMILES_TOKENIZER

# Set Polars to use maximum number of threads automatically
os.environ["POLARS_MAX_THREADS"] = str(multiprocessing.cpu_count())
os.environ["POLARS_VERBOSE"] = "1"


def tokenize_smiles(smiles: str) -> str:
    tokenized = SMILES_TOKENIZER(
        smiles,
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    return (tokenized["input_ids"], tokenized["attention_mask"])


def parallel_tokenize(smiles_list):
    with multiprocessing.Pool() as pool:
        return pool.map(tokenize_smiles, smiles_list)


# Function to read the cached CID-SMILES file, process it, and write output
def read_cached_CID_smiles_in_batches(batch_size=1_999_968):
    # Read data in batches
    reader = pl.read_parquet("CID-SMILES-only-canonical.parquet")

    logger.info(f"Processing data in batches of {batch_size}")

    # Process the data in batches
    for batch_index, batch in enumerate(reader.collect().iter_slices(batch_size)):
        smiles_batch = batch["smiles"].to_list()
        # Parallel canonicalization
        tokenized = parallel_tokenize(smiles_batch)
        # Add canonical SMILES to the dataframe
        batch_ = batch.with_columns([pl.Series(tokenized).alias("tokens")])

        # Save batch as a parquet file
        batch_.write_parquet(f"pubchem_canonical_tokenized/CID-SMILES-tokenized-batch-{batch_index}.parquet")
        logger.info(f"Batch {batch_index} processed and saved.")

    logger.info("Processing completed and data saved.")


if __name__ == "__main__":
    read_cached_CID_smiles_in_batches()
