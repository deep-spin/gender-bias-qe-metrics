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
    parser.add_argument("--output_dir", type=str,  help="Output directory",default="./results/scores/nonambiguous-counterfactual/mtgeneval/gt_translations")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    parser.add_argument("--translations_dir", type=str, default='./data/mtgeneval/google_translated_counterfactual/', help="Path containing the translations to score. (GT translations in our case)")
    parser.add_argument("--dry_run", action="store_true", help="Dry run for testings")
    return parser.parse_args()



def read_mtgeneval_counterfactual_dataset_with_translations(lang: str):
    dataset_name = "gsarti/mt_geneval"
    dataset_config = f"sentences_en_{lang}"
    data = load_dataset(dataset_name, dataset_config, split="test")
    logger.info(f"Loaded {len(data)} entries")
    logger.info(f"Sample entry: {data[0]}")
    if args.dry_run:
        data = data.select(range(5))
    return data


def read_translations(translations_dir: str,lang: str,dry_run: bool = False):
    translations_path = os.path.join(translations_dir,f"mt_geneval_sentences_en_{lang}.csv") # we have saved in these csv files the translations of the MT-GenEval dataset using GT (+ other details that are used in the notebooks.)
    translations_df = pd.read_csv(translations_path,header=0)
    if dry_run:
        translations_df = translations_df.iloc[:5]
    return translations_df

def construct_metric_inputs(lang: str,translations_dir: str,dry_run: bool = False):
    data = read_mtgeneval_counterfactual_dataset_with_translations(lang)
    translations_df = read_translations(translations_dir,lang,dry_run)
    metric_inputs = {}
    for stype in ["feminine", "masculine"]:
        for rtype in ["feminine", "masculine"]:
            metric_inputs[f"{stype}_{rtype}"] = [
                {"src": src, "mt": mt}
                for src, mt in zip(data[f"source_{stype}"], translations_df[f"translate_{rtype}"])
            ]
    return data,translations_df,metric_inputs

    


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
    data,translations_df,metric_inputs = construct_metric_inputs(args.tgt_lang,args.translations_dir,args.dry_run)
    
    scores_dict = {}
    model_name = args.qe_metric_name_or_path
    logger.info(f"Metric used: {model_name} !!!")

    for key, input in metric_inputs.items():
        # run metric
        
        scores = run_metric(model_name,input)
        scores_dict[key] = scores


    # save results
    df = translations_df.copy()
    df = df[['orig_id', 'source_feminine', 'translate_feminine', 'source_masculine', 'translate_masculine']]
    df = df.rename(columns={'orig_id': 'ID'})
    
    for key, scores in scores_dict.items():
        df[f"score_{key}"] = scores
    
    # save results
    model_name = model_name.replace("/", "--")
    output_dir = os.path.join(args.output_dir, f"{args.tgt_lang}/{model_name}")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"scores.csv"), index=False)


if __name__ == "__main__":
    args = get_args()
    main(args)
