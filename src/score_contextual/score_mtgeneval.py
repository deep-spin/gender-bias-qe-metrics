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
    parser.add_argument("--output_dir", type=str,  help="Output directory",default="./results/scores/nonambiguous-contextual/mtgeneval")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    parser.add_argument("--context_translations_path", default=None, type=str, help="path to the file containing the translated contexts (only for neural metrics)")
    parser.add_argument("--translation_model_name", default=None, type=str, help="Model name used for translating the contexts. It is used only for saving the results to the output directory. (only for neural metrics)")
    parser.add_argument("--gemba_src_ctx_type", default="standard", type=str,choices=["standard", "paraphrased"], help="Gemba prompt type for the LLM metrics.")
    parser.add_argument("--dry_run", action="store_true", help="Dry run for testings")
    parser.add_argument("--metric_type", type=str, required=True, choices=["neural", "llm"])
    return parser.parse_args()



def read_mtgeneval_contextual_dataset_original_context(lang: str,metric_type: str):
    dataset_name = "gsarti/mt_geneval"
    dataset_config = f"context_en_{lang}"
    data = load_dataset(dataset_name, dataset_config, split="test")
    logger.info(f"Loaded {len(data)} entries")
    logger.info(f"Sample entry: {data[0]}")
    if args.dry_run:
        data = data.select(range(5))

    if metric_type == "neural":
        srcs = data["source"]
        o_input = [
                {"src": f"{context} {src}", "mt": f"{context} {mt}"} for src, mt, context in zip(srcs, data["reference_original"],data["context"])
            ]
        f_input = [
            {"src": f"{context} {src}", "mt": f"{context} {mt}"} for src, mt, context in zip(srcs, data["reference_flipped"],data["context"])
        ]
        return data,o_input, f_input
    else:
        # we have the gemba case as we have an LLM model. In this case inputs are formed differently.
        srcs = data["source"]
        o_input = [
                {"src": f"{src}", "mt": f"{mt}", "src_ctx": f"{context}"} for src, mt, context in zip(srcs, data["reference_original"],data["context"])
            ]
        f_input = [
            {"src": f"{src}", "mt": f"{mt}", "src_ctx": f"{context}"} for src, mt, context in zip(srcs, data["reference_flipped"],data["context"])
        ]
        return data,o_input, f_input


def read_mtgeneval_contextual_dataset_translated_context(lang: str,context_translations_path: str):
    dataset_name = "gsarti/mt_geneval"
    dataset_config = f"context_en_{lang}"
    data = load_dataset(dataset_name, dataset_config, split="test")
    logger.info(f"Loaded {len(data)} entries")
    logger.info(f"Sample entry: {data[0]}")
    if args.dry_run:
        data = data.select(range(5))

    srcs = [f"{context} {src}" for context, src in zip(data["context"], data["source"])]

    # load translated context (this is used only for the neural metrics)
    with open(context_translations_path, "r") as f:
        translated_contexts = f.readlines()
    translated_contexts = [context.strip() for context in translated_contexts]

    o_input = [
            {"src": src, "mt": f"{translated_contexts[i]} {mt}"} for i, (src, mt) in enumerate(zip(srcs, data["reference_original"]))
        ]
    f_input = [
        {"src": src, "mt": f"{translated_contexts[i]} {mt}"} for i, (src, mt) in enumerate(zip(srcs, data["reference_flipped"]))
    ]
    return data,o_input, f_input



def run_metric_neural(model_name: str, orig_input: List[Dict], flipped_input: List[Dict]):
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
    else:
        logger.error(f"Model name {model_name} not supported.")
        exit()


def run_metric_llm(model_name: str, orig_input: List[Dict], flipped_input: List[Dict],gemba_src_ctx_type: str):
    if gemba_src_ctx_type not in ["standard", "paraphrased"]:
        logger.error(f"Gemba src ctx type {gemba_src_ctx_type} not supported. Please use standard or paraphrased.")
        exit()
    logger.info(f"Running LLM metric: {model_name} with gemba src ctx type: {gemba_src_ctx_type}")
    if 'gpt' in model_name.lower():
        logger.error(f"GPT models are not supported in this script. Please use the openai_batch_mtgeneval.py script.")
        exit()
    if 'mistral' or 'gemma' or 'llama' in model_name.lower():
        scorer = GembaModelScorer(model_name, "en", args.tgt_lang, gemba_use_src_context=True, gemba_src_ctx_type=gemba_src_ctx_type, gemba_num_shots=0)
        orig_scores, orig_scores_numeric = scorer.score(orig_input, extract_scores=True)
        flipped_scores, flipped_scores_numeric = scorer.score(flipped_input, extract_scores=True)
        return orig_scores_numeric, flipped_scores_numeric
    else:
        logger.error(f"Model name {model_name} not supported.")
        exit()



def main(args):

    # read dataset
    if args.context_translations_path is not None:
        if args.translation_model_name is None:
            logger.warning("Translation model name not provided. Please provide the model name used to translate the contexts. Used only for saving the results to the output directory.")
            translation_model_name = "dummy-model"
        else:
            translation_model_name = args.translation_model_name
        if args.metric_type == "llm":
            logger.error(f"LLM models are not supported for the translated context case. Please use the neural metrics.")
            exit()
        data,orig_input, flipped_input = read_mtgeneval_contextual_dataset_translated_context(args.tgt_lang,args.context_translations_path)
    else:
        data,orig_input, flipped_input = read_mtgeneval_contextual_dataset_original_context(args.tgt_lang,args.metric_type)
    
    # run metric
    model_name = args.qe_metric_name_or_path
    logger.info(f"Metric used: {model_name} !!!")
    if args.metric_type == "neural":
        orig_scores, flipped_scores = run_metric_neural(model_name,orig_input, flipped_input)
    else:
        # use of gemba scorer for the llm metrics
        orig_scores, flipped_scores = run_metric_llm(model_name,orig_input, flipped_input,args.gemba_src_ctx_type)
    
    

    # save results
    df = data.to_pandas()
    df = df.rename(columns={'orig_id': 'ID'})
    final_df = df.set_index("ID")
    final_df["score_Original"] = orig_scores
    final_df["score_Flipped"] = flipped_scores

    # save results
    model_name = model_name.replace("/", "--")
    if args.metric_type == "neural":
        if args.context_translations_path is not None:
            output_dir = os.path.join(args.output_dir, f"{args.tgt_lang}/{model_name}/with-translated-contexts/{translation_model_name}")
        else:
            output_dir = os.path.join(args.output_dir, f"{args.tgt_lang}/{model_name}/with-original-contexts/")
    else:
        # in this case we have an llm model. We need to save the results in the output directory...
        output_dir = os.path.join(args.output_dir, f"{args.tgt_lang}/{model_name}/{args.gemba_src_ctx_type}")

    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f"scores.csv"), index=True)


if __name__ == "__main__":
    args = get_args()
    main(args)