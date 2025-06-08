import json
import pickle as pkl
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
from gafuncs import CachedBatchFunction  # Assuming this custom module is available

# Use the SAME helper and SimpleMoleculeAnalyzer's PKL config definition
from generate_molformula_files import SimpleMoleculeAnalyzer, get_1d_target_embedding_from_raw_batches_pkl
from loguru import logger
from mol_ga.graph_ga.gen_candidates import graph_ga_blended_generation
from mol_ga.preconfigured_gas import default_ga
from prune_sim import gpu_encode_smiles_variable, load_models_dict, tanimoto_similarity
from rdkit import Chem  # For atom counts
from torch.nn.functional import cosine_similarity as torch_cosine_similarity
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def compute_individual_atom_counts(individual: str) -> dict | None:
    mol = Chem.MolFromSmiles(individual)
    if not mol:
        logger.warning(f"Invalid SMILES for atom count: {individual}")
        return None
    mol = Chem.AddHs(mol)
    counts = {}
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
    return counts


def calculate_mf_penalty(smi: str, atom_counts_orig: dict) -> float:
    """Calculates the molecular formula penalty for a single SMILES string."""
    counts_i = compute_individual_atom_counts(smi)
    if not counts_i:
        return -1000.0  # Heavy penalty for invalid SMILES

    total_orig_atoms = sum(atom_counts_orig.values()) if atom_counts_orig else 1.0
    if total_orig_atoms == 0:
        total_orig_atoms = 1.0

    penalty = sum(abs(counts_i.get(s, 0) - atom_counts_orig.get(s, 0)) for s in set(counts_i) | set(atom_counts_orig))
    return -penalty / total_orig_atoms


def reward_function_ga(individuals: list[str], ga_models: dict, target_1D_embs: dict, atom_counts_orig: dict):
    if not individuals:
        return np.array([])
    cand_smiles_embs = gpu_encode_smiles_variable(individuals, ga_models)  # Dict: mod -> (N,D)

    mf_loss = np.array([calculate_mf_penalty(smi, atom_counts_orig) for smi in individuals])

    scores_all_mods_np, num_ok_mods = [], 0
    for spec, target_emb_1D_gpu in target_1D_embs.items():  # target_emb is 1D (D,)
        if spec not in cand_smiles_embs or cand_smiles_embs[spec] is None or cand_smiles_embs[spec].nelement() == 0:
            continue  # No candidate embeddings for this modality

        cand_embs_mod_gpu = cand_smiles_embs[spec]  # (N, D_mod)

        if target_emb_1D_gpu.ndim != 1 or cand_embs_mod_gpu.shape[1] != target_emb_1D_gpu.shape[0]:
            logger.warning(
                f"Reward: Dim mismatch for {spec}. Target: {target_emb_1D_gpu.shape}, Cand: {cand_embs_mod_gpu.shape}. Skipping."
            )
            continue

        sims_gpu = torch_cosine_similarity(
            target_emb_1D_gpu.unsqueeze(0).to("cuda"), cand_embs_mod_gpu.to("cuda"), dim=1
        )  # (1,D) vs (N,D) -> (N,)
        scores_all_mods_np.append(sims_gpu.cpu().numpy())
        num_ok_mods += 1

    if num_ok_mods == 0:
        return mf_loss  # Only MF penalty if no spectral scores

    avg_cosine_sim = np.mean(np.array(scores_all_mods_np), axis=0)
    return avg_cosine_sim + mf_loss


def calculate_detailed_scores(smi: str, models: dict, target_1D_embs: dict, atom_counts_orig: dict) -> dict:
    """Calculates and returns a dictionary of detailed scores for a single SMILES."""
    if smi == "N/A" or not smi:
        return {"mf_penalty": -1000.0}

    scores = {}
    # 1. Calculate Molecular Formula Penalty
    scores["mf_penalty"] = calculate_mf_penalty(smi, atom_counts_orig)

    # 2. Calculate Modality-specific Cosine Similarities
    smi_emb_dict = gpu_encode_smiles_variable([smi], models)
    for spec, target_emb in target_1D_embs.items():
        score_key = f"{spec}_cosine_sim"
        if spec not in smi_emb_dict or smi_emb_dict[spec] is None or smi_emb_dict[spec].nelement() == 0:
            scores[score_key] = 0.0
            continue

        cand_emb = smi_emb_dict[spec].to("cuda")  # (1, D)
        target_emb = target_emb.to("cuda")  # (D,)

        sim = torch_cosine_similarity(target_emb.unsqueeze(0), cand_emb, dim=1)
        scores[score_key] = sim.item()

    return scores


def run_ga_instance(
    initial_df: pd.DataFrame,
    models: dict,
    idx: int,
    orig_smi: str,
    atom_counts_orig: dict,
    target_1D_embs: dict,
    out_dir: Path,
    ga_params: dict,
):
    # Unpack GA parameters
    init_pop = ga_params["initial_population_size_from_pruning"]
    gens = ga_params["generations"]
    offspring = ga_params["offspring_size"]
    pop_ga = ga_params["population_size"]
    seed_val = ga_params["seed"]
    frac_mutate = ga_params["frac_graph_ga_mutate"]

    sort_key = next(
        (
            k
            for k in ["sum_of_all_individual_similarities", "similarity"]
            if k in initial_df.columns and not initial_df[k].isna().all()
        ),
        None,
    )
    top_n_smiles = (
        initial_df.sort_values(by=sort_key, ascending=False)["canonical_smiles"].head(init_pop).tolist()
        if sort_key
        else initial_df["canonical_smiles"].head(init_pop).tolist()
    )

    reward_f = partial(reward_function_ga, ga_models=models, target_1D_embs=target_1D_embs, atom_counts_orig=atom_counts_orig)

    ga_logger = logger

    ga_res = default_ga(
        starting_population_smiles=top_n_smiles,
        scoring_function=CachedBatchFunction(reward_f, original_smiles=orig_smi),
        max_generations=gens,
        offspring_size=offspring,
        population_size=pop_ga,
        logger=ga_logger,
        rng=random.Random(seed_val),
        offspring_gen_func=partial(graph_ga_blended_generation, frac_graph_ga_mutate=frac_mutate),
    )

    best_sc, best_smi = max(ga_res.population, key=lambda x: x[0]) if ga_res.population else (-float("inf"), "N/A")
    logger.info(f"GA Best for idx {idx}: {best_smi} (score {best_sc:.4f})")

    # Calculate detailed scores for the best individual
    detailed_scores = calculate_detailed_scores(best_smi, models, target_1D_embs, atom_counts_orig)

    summary_data = {
        "target_idx": idx,
        "original_smiles": orig_smi,
        "best_ga_smiles": best_smi,
        "best_ga_score": float(best_sc),
        "tanimoto_to_original": tanimoto_similarity(best_smi, orig_smi) if best_smi != "N/A" else 0.0,
        "best_ga_detailed_scores": detailed_scores,
        "ga_modalities_scored": list(target_1D_embs.keys()),
        "initial_population_size": len(top_n_smiles),
        "ga_parameters": ga_params,  # Store all GA parameters for reproducibility
    }
    with (out_dir / f"{idx}_ga_summary.json").open("w") as f:
        json.dump(summary_data, f, indent=2)

    # The `ga_res` object contains the full history of the run.
    # You can access it via `ga_res.history`, which is a list of tuples:
    # [(generation_0, population_0), (generation_1, population_1), ...]
    # where population is a list of (score, smiles) tuples.
    # Saving the whole object with pickle preserves this history.
    with (out_dir / f"{idx}_ga_details.pkl").open("wb") as f:
        pkl.dump(ga_res, f)
    return ga_res


def cli(
    start_range: int,
    end_range: int,
    seed: int = 42,
    configs_path: str = "../../configs",
    dataset_path: str = "../../data/test_split.parquet",
    # GA: model experiments & RAW target embedding file paths (List[BATCH Dict])
    ga_ir_exp: str | None = "test/ir",
    ga_cnmr_exp: str | None = "test/cnmr_pretrain",
    ga_hnmr_exp: str | None = "test/hnmr_augment_pretrain",
    ga_hsqc_exp: str | None = "test/hsqc",
    # GA: RAW embedding file paths
    ga_ir_raw_emb_path: str | None = None,
    ga_cnmr_raw_emb_path: str | None = None,
    ga_hnmr_raw_emb_path: str | None = None,
    ga_hsqc_raw_emb_path: str | None = None,
    # GA parameters
    init_pop_ga: int = 512,
    gens_ga: int = 50,
    offspring_ga: int = 2048,
    pop_ga: int = 256,
    base_cache_dir: str = "four_modalities_ga_cache",
):
    ga_model_exps = {"ir": ga_ir_exp, "cnmr": ga_cnmr_exp, "hnmr": ga_hnmr_exp, "hsqc": ga_hsqc_exp}
    ga_raw_emb_paths_map = {
        "ir": ga_ir_raw_emb_path,
        "cnmr": ga_cnmr_raw_emb_path,
        "hnmr": ga_hnmr_raw_emb_path,
        "hsqc": ga_hsqc_raw_emb_path,
    }

    # --- Collect all GA parameters for logging ---
    ga_params = {
        "seed": seed,
        "initial_population_size_from_pruning": init_pop_ga,
        "generations": gens_ga,
        "offspring_size": offspring_ga,
        "population_size": pop_ga,
        "frac_graph_ga_mutate": 0.1,  # Hardcoded in the original script
        "model_experiments": {k: v for k, v in ga_model_exps.items() if v},
        "raw_embedding_paths": {k: v for k, v in ga_raw_emb_paths_map.items() if v},
        "dataset_path": dataset_path,
        "base_cache_dir": base_cache_dir,
    }

    ga_models_for_scoring = load_models_dict(configs_path, ga_model_exps)
    active_ga_model_modalities = [m for m, model in ga_models_for_scoring.items() if model]
    if not active_ga_model_modalities:
        logger.error("No GA models loaded. Exiting.")
        return
    logger.info(f"GA scoring models: {active_ga_model_modalities}")

    analyzer_user_active_spectra = [m for m in SimpleMoleculeAnalyzer.ALL_SPECTRA_TYPES if locals().get(f"analyzer_{m}_exp")]
    analyzer = SimpleMoleculeAnalyzer(
        models_config_path=configs_path,
        results_dir=str(Path(base_cache_dir) / "analyzer_candidates_output"),
        cache_dir=str(Path(base_cache_dir) / "analyzer_process_cache"),
        active_spectra=analyzer_user_active_spectra,
    )
    analyzer.load_models(
        ir_experiment=ga_ir_exp, cnmr_experiment=ga_cnmr_exp, hnmr_experiment=ga_hnmr_exp, hsqc_experiment=ga_hsqc_exp
    )
    analyzer.load_data(
        dataset_path,
        ir_embeddings_path=ga_ir_raw_emb_path,
        cnmr_embeddings_path=ga_cnmr_raw_emb_path,
        hnmr_embeddings_path=ga_hnmr_raw_emb_path,
        hsqc_embeddings_path=ga_hsqc_raw_emb_path,
    )
    if not analyzer.available_modalities:
        logger.error("Analyzer not ready (no available modalities after loading models/data). Exiting.")
        return
    logger.info(f"Analyzer ready with modalities: {analyzer.available_modalities}. Summary: {analyzer.get_summary()}")

    try:
        main_dataset_df = pd.read_parquet(dataset_path)
    except Exception as e:
        logger.error(f"Cannot load main dataset: {dataset_path} - {e}")
        return

    for i in range(int(start_range), int(end_range)):
        target_idx = i
        logger.info(f"--- GA for Target Index: {target_idx} ---")
        if not (0 <= target_idx < len(main_dataset_df)):
            logger.error(f"Idx {target_idx} OOB for main dataset.")
            continue

        original_smiles = main_dataset_df.smiles.iloc[target_idx]
        atom_counts_orig = compute_individual_atom_counts(original_smiles)
        if not atom_counts_orig:
            logger.error(f"No atom counts for {original_smiles}. Skipping GA for this index.")
            continue

        initial_candidates_df = analyzer.process_molecular_formula(smiles_index=target_idx)
        if initial_candidates_df.empty:
            logger.warning(f"No initial candidates from Analyzer for idx {target_idx}. Skipping GA for this index.")
            continue

        ga_target_1D_embs = {}
        for spec_type in active_ga_model_modalities:  # Only prepare targets for which GA has a scoring model
            raw_path = ga_raw_emb_paths_map.get(spec_type)
            pickle_config = SimpleMoleculeAnalyzer.RAW_EMBEDDING_PKL_CONFIGS.get(spec_type)
            if not raw_path or not pickle_config:
                continue

            model_device = "cuda" if ga_models_for_scoring[spec_type] else "cpu"
            emb = get_1d_target_embedding_from_raw_batches_pkl(
                raw_path, target_idx, pickle_config, len(main_dataset_df), model_device
            )
            if emb is not None:
                ga_target_1D_embs[spec_type] = emb

        if not ga_target_1D_embs:
            logger.error(f"No 1D target embeddings prepared for GA reward (idx {target_idx}). Skipping GA for this index.")
            continue

        final_ga_models_to_use = {m: ga_models_for_scoring[m] for m in ga_target_1D_embs if m in ga_models_for_scoring}
        if not final_ga_models_to_use:
            logger.error(f"No GA models align with prepared target embeddings for idx {target_idx}. Skipping GA.")
            continue

        ga_run_dir = Path(base_cache_dir) / f"idx_{target_idx}_ga_run"
        ga_run_dir.mkdir(parents=True, exist_ok=True)
        run_ga_instance(
            initial_df=initial_candidates_df,
            models=final_ga_models_to_use,
            idx=target_idx,
            orig_smi=original_smiles,
            atom_counts_orig=atom_counts_orig,
            target_1D_embs=ga_target_1D_embs,
            out_dir=ga_run_dir,
            ga_params=ga_params,  # Pass the comprehensive params dictionary
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("All specified GA runs complete.")


if __name__ == "__main__":
    import fire

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
    fire.Fire(cli)
