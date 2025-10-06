import argparse
import os
import json
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
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Name of the QE metric to use")
    parser.add_argument("--hypothesis_path", type=str,  help="File with generated hypotheses for reranking.",required=True)
    parser.add_argument("--output_path", type=str,  help="File to save the best generations.",required=True)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the metric scorer.")
    return parser.parse_args()


def main(args):

    # read hypotheses
    with open(args.hypothesis_path, "r", encoding="utf-8") as f:
        grouped_generations = json.load(f)
    
    # run metric
    
    # load qad-reranking model
    logger.info(f"To run metric scorer: {args.model_name_or_path}")
    metric_scorer = COMETScorer(args.model_name_or_path)
    num_return_sequences = len(grouped_generations[0]["outputs"])

    # form inputs for metric as list of dicts with src and mt by extending the outputs
    inputs = [
        {
            "src": grouped_generations[i]["input"],
            "mt": grouped_generations[i]["outputs"][j]
        }
        for i in range(len(grouped_generations))
        for j in range(num_return_sequences)
    ]
    scores = metric_scorer.score(inputs,batch_size=args.batch_size)["scores"]
    # reform scores as list of lists 
    scores_reformatted = [scores[i:i+num_return_sequences] for i in range(0, len(scores), num_return_sequences)]
    # return the best scoring output for each src
    best_generations = [{"input": grouped_generations[i]["input"], "outputs": [grouped_generations[i]["outputs"][np.argmax(scores_reformatted[i])]]} for i in range(len(grouped_generations))]
    # save best generations
    logger.info(f"Saving best generations to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(best_generations, f,indent=4)



if __name__ == "__main__":
    args = get_args()
    main(args)