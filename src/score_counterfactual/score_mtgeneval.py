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
    parser.add_argument("--output_dir", type=str,  help="Output directory",default="./results/scores/nonambiguous-counterfactual/mtgeneval")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    parser.add_argument("--dry_run", action="store_true", help="Dry run for testings")
    return parser.parse_args()



def read_mtgeneval_counterfactual_dataset(lang: str):
    dataset_name = "gsarti/mt_geneval"
    dataset_config = f"sentences_en_{lang}"
    data = load_dataset(dataset_name, dataset_config, split="test")
    logger.info(f"Loaded {len(data)} entries")
    logger.info(f"Sample entry: {data[0]}")
    if args.dry_run:
        data = data.select(range(5))


    options = ["feminine", "masculine"]
    metric_inputs = {}
    for stype in options:
        for rtype in options:
            metric_inputs[f"{stype}_{rtype}"] = [
                {"src": src, "mt": mt}
                for src, mt in zip(
                    data[f"source_{stype}"], data[f"reference_{rtype}"]
                )
            ]
    return data,metric_inputs


def run_metric(model_name: str, input: List[Dict]):
    if "comet" in model_name.lower():
        scorer = COMETScorer(model_name)
        scores = np.array(scorer.score(input)["scores"])
        return scores
    elif "metricx" in model_name.lower():
        scorer = MetricXScorer(model_name)
        scores = np.array(scorer.score(input))
        return scores
    elif "gpt" in model_name.lower():
        logger.warning(f"Not implemented in this script. For GPT, please use the openai_batch_mtgeneval.py script.")
        exit()
    else:
        scorer = GembaModelScorer(model_name, "en", args.tgt_lang, gemba_use_src_context=False, gemba_src_ctx_type="standard", gemba_num_shots=0)
        scores, scores_numeric = scorer.score(input, extract_scores=True)
        return scores_numeric


def main(args):

    # read dataset
    data,metric_inputs = read_mtgeneval_counterfactual_dataset(args.tgt_lang)
    
    scores_dict = {}
    model_name = args.qe_metric_name_or_path
    logger.info(f"Metric used: {model_name} !!!")

    for key, input in metric_inputs.items():
        # run metric
        
        scores = run_metric(model_name,input)
        scores_dict[key] = scores


    # save results
    df = data.to_pandas()
    df = df.rename(columns={'orig_id': 'ID'})
    final_df = df.set_index("ID")
    for key, scores in scores_dict.items():
        final_df[f"score_{key}"] = scores
    
    drop_columns = ['source_feminine_annotated','reference_feminine_annotated','source_masculine_annotated','reference_masculine_annotated','source_feminine_keywords','reference_feminine_keywords','source_masculine_keywords','reference_masculine_keywords']
    final_df.drop(columns=drop_columns, inplace=True)
    breakpoint()

    # save results
    model_name = model_name.replace("/", "--")
    output_dir = os.path.join(args.output_dir, f"{args.tgt_lang}/{model_name}")
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f"scores.csv"), index=True)


if __name__ == "__main__":
    args = get_args()
    main(args)
