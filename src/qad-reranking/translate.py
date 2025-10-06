import argparse
import random
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from src.utils_metrics import apply_template_tower, write_lines
from datasets import load_dataset

from src.utils_metrics import COMETScorer


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model to use for translation.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory for saving generations.")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language.")
    parser.add_argument("--dry_run", action="store_true", required=False, default=False, help="Dry run for testing. Will only translate 5 rows.")

    # generation args
    parser.add_argument("--topk", type=int, default=-1, help="Top-k decoding.")
    parser.add_argument("--epsilon_cutoff", type=float, default=0.0)
    parser.add_argument("--num_return_sequences", type=int, required=False, default=1) # number of translations to generate for each source (used for reranking)!
    parser.add_argument("--temperature", type=float, required=False, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, required=False, default=1024)

    # qad-reranking args
    # parser.add_argument("--qad_reranking_model_name_or_path", type=str, default=None, help="Path to the qad-reranking model to use for translation.")
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size for running the metric scorer.")
    return parser.parse_args()


def qad_reranking(metric_name_or_path: str, grouped_generations: list[dict], batch_size: int):
    # load qad-reranking model
    metric_scorer = COMETScorer(metric_name_or_path)
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
    scores = metric_scorer.score(inputs,batch_size=batch_size)["scores"]
    # reform scores as list of lists 
    scores_reformatted = [scores[i:i+num_return_sequences] for i in range(0, len(scores), num_return_sequences)]
    # return the best scoring output for each src
    best_generations = [grouped_generations[i]["outputs"][np.argmax(scores_reformatted[i])] for i in range(len(grouped_generations))]
    breakpoint()
    return best_generations


def main(
    args: argparse.Namespace
):
    # load dataset with contexts
    data = load_dataset("gsarti/mt_geneval", f"context_en_{args.tgt_lang}", split="test")
    logger.info(f"Loaded dataset {data}")
    logger.info(f"Len: {len(data)}")
    logger.info(f"Random row: {data[random.randint(0, len(data))]}")

    if args.dry_run:
        logger.info(f"Running dry run for 5 rows")
        data = data.select(range(5))

    data = data["source"]
    logger.info(f"Sample source to translate: {data[0]}")

    # vllm usage
    # initialize model and run generation
    model = LLM(
        model=args.model_name_or_path,
        seed=42,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    
    s = SamplingParams(n=args.num_return_sequences, max_tokens=args.max_new_tokens, temperature=args.temperature, min_p=args.epsilon_cutoff,top_k=args.topk, stop = ["\n", "\\n", "</s>"],)
    tokenizer = model.get_tokenizer()

    # apply template to source for Instruct
    if "Instruct" in args.model_name_or_path:
        srcs = [tokenizer.apply_chat_template(
            [{"role": "user", "content": apply_template_tower(x, "en", args.tgt_lang)}], tokenize=False) for x in data]
    else:
        srcs = data
    
    # generate candidates
    candidates = []
    for l in srcs:
        candidates.extend([l])  
    model_output = model.generate(candidates, s, use_tqdm=True)
    grouped_generations = [
        {
            'input': data[i],
            'outputs': [o.text for o in output.outputs]
        }
        for i, output in  enumerate(model_output)
    ]

    # save generations to file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(grouped_generations, f,indent=4)

    # if args.qad_reranking_model_name_or_path is not None:
    #     best_generations = qad_reranking(args.qad_reranking_model_name_or_path,grouped_generations,args.batch_size)
    # else:
    #     best_generations = [output.outputs[0].text for output in model_output]
        
    # write generations to file
    # write_lines(args.output_path, best_generations)


if __name__ == "__main__":
    args = parse_args()
    main(args)