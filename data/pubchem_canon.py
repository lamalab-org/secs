import contextlib
import multiprocessing
import os

import polars as pl
from loguru import logger
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Set Polars to use maximum number of threads automatically
os.environ["POLARS_MAX_THREADS"] = str(multiprocessing.cpu_count())
os.environ["POLARS_VERBOSE"] = "1"


# Function to canonicalize SMILES
def canonicalize_smiles(smiles: str) -> str:
    with contextlib.suppress(Exception):
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    return None


# Canonicalize SMILES in parallel
def parallel_canonicalize(smiles_list):
    with multiprocessing.Pool() as pool:
        return pool.map(canonicalize_smiles, smiles_list)


# Function to read the cached CID-SMILES file, process it, and write output
def read_cached_CID_smiles_in_batches(batch_size=1_999_968):
    # Read data in batches
    reader = pl.scan_csv("CID-SMILES", separator="\t", has_header=False)

    # Rename columns
    reader = reader.rename({"column_1": "CID", "column_2": "smiles"})

    logger.info(f"Processing data in batches of {batch_size}")

    # Process the data in batches
    for batch_index, batch in enumerate(reader.collect().iter_slices(batch_size)):
        smiles_batch = batch["smiles"].to_list()
        # Parallel canonicalization
        canonical_smiles = parallel_canonicalize(smiles_batch)
        # Add canonical SMILES to the dataframe
        batch_ = batch.with_columns(
            [pl.Series(canonical_smiles).alias("canonical_smiles")]
        )

        # Save batch as a parquet file
        batch_.write_parquet(
            f"pubchem_canonical/CID-SMILES-canonical-batch-{batch_index}.parquet"
        )
        logger.info(f"Batch {batch_index} processed and saved.")

    logger.info("Processing completed and data saved.")


if __name__ == "__main__":
    read_cached_CID_smiles_in_batches()
