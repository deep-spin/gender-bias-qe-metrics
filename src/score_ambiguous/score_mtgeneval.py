import argparse
import os
import pandas as pd
import numpy as np
from typing import List, Dict
from datasets import load_dataset

from src.utils_metrics import COMETScorer, GembaModelScorer, MetricXScorer, OpenAIModelScorer

from loguru import logger
import sys
# Configure loguru with explicit color settings
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,  # Explicitly enable colors
    backtrace=True,
    diagnose=True,
    enqueue=True
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qe_metric_name_or_path", type=str, required=True, help="Name of the QE metric to use")
    parser.add_argument("--output_dir", type=str,  help="Output directory",default="./results/scores/ambiguous/mtgeneval")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    parser.add_argument("--dry_run", action="store_true", help="Dry run for testings")
    return parser.parse_args()



def read_mtgeneval_contextual_dataset(lang: str):
    dataset_name = "gsarti/mt_geneval"
    dataset_config = f"context_en_{lang}"
    data = load_dataset(dataset_name, dataset_config, split="test")
    logger.info(f"Loaded {len(data)} entries")
    logger.info(f"Sample entry: {data[0]}")
    if args.dry_run:
        data = data.select(range(5))

    srcs = data["source"]
    o_input = [
            {"src": src, "mt": mt} for src, mt in zip(srcs, data["reference_original"])
        ]
    f_input = [
        {"src": src, "mt": mt} for src, mt in zip(srcs, data["reference_flipped"])
    ]
    return data,o_input, f_input


def run_metric(model_name: str, orig_input: List[Dict], flipped_input: List[Dict]):
    if "comet" in model_name.lower():
        scorer = COMETScorer(model_name)
        orig_scores = np.array(scorer.score(orig_input)["scores"])
        flipped_scores = np.array(scorer.score(flipped_input)["scores"])
        return orig_scores, flipped_scores
    elif "metricx" in model_name.lower():
        scorer = MetricXScorer(model_name)
        orig_scores = np.array(scorer.score(orig_input))
        flipped_scores = np.array(scorer.score(flipped_input))
        return orig_scores, flipped_scores
    elif "gpt" in model_name.lower():
        logger.warning(f"Not implemented in this script. For GPT, please use the openai_batch_mgente.py script.")
        exit()
    else:
        scorer = GembaModelScorer(model_name, "en", args.tgt_lang, gemba_use_src_context=False, gemba_src_ctx_type="standard", gemba_num_shots=0)
        orig_scores, orig_scores_numeric = scorer.score(orig_input, extract_scores=True)
        flipped_scores, flipped_scores_numeric = scorer.score(flipped_input, extract_scores=True)
        return orig_scores_numeric, flipped_scores_numeric


def main(args):

    # read dataset
    data,orig_input, flipped_input = read_mtgeneval_contextual_dataset(args.tgt_lang)
    
    # run metric
    model_name = args.qe_metric_name_or_path
    logger.info(f"Metric used: {model_name} !!!")
    orig_scores, flipped_scores = run_metric(model_name,orig_input, flipped_input)

    # save results
    df = data.to_pandas()
    df = df.rename(columns={'orig_id': 'ID'})
    final_df = df.set_index("ID")
    final_df["score_Original"] = orig_scores
    final_df["score_Flipped"] = flipped_scores

    # save results
    model_name = model_name.replace("/", "--")
    output_dir = os.path.join(args.output_dir, f"{args.tgt_lang}/{model_name}")
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f"scores.csv"), index=True)


if __name__ == "__main__":
    args = get_args()
    main(args)