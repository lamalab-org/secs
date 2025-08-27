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
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        return pool.map(tokenize_smiles, smiles_list)


# Function to read the cached CID-SMILES file, process it, and write output
def read_cached_CID_smiles_in_batches(batch_size=1_999_968):
    # Read data in batches
    reader = pl.scan_parquet("/home/mirzaa/projects/MoleculeBind/app/cache_pubchem.parquet")

    logger.info(f"Processing data in batches of {batch_size}")

    # Process the data in batches
    for batch_index, batch in enumerate(reader.collect().iter_slices(batch_size)):
        # check if file is already processed
        if os.path.exists(f"/data/mirzaa/pubchem_canonical_tokenized/CID-SMILES-tokenized-batch-{batch_index}.parquet"):
            logger.info(f"Batch {batch_index} already processed, skipping.")
            continue

        smiles_batch = batch["smiles"].to_list()
        # Parallel tokenize the SMILES strings
        tokenized = parallel_tokenize(smiles_batch)
        # Add canonical SMILES to the dataframe
        batch_ = batch.with_columns([pl.Series(tokenized).alias("tokens")])

        # Save batch as a parquet file
        batch_.write_parquet(f"/data/mirzaa/pubchem_canonical_tokenized/CID-SMILES-tokenized-batch-{batch_index}.parquet")
        logger.info(f"Batch {batch_index} processed and saved.")

    logger.info("Processing completed and data saved.")


if __name__ == "__main__":
    read_cached_CID_smiles_in_batches()
