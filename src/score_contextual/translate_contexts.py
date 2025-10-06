import argparse
import logging
import random
import os
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from src.utils_metrics import apply_template_tower, write_lines
from datasets import load_dataset


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
    parser.add_argument("--output_dir", type=str, default="./data/mtgeneval/translated-contexts", help="Path to the output directory.")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language.")
    parser.add_argument("--dry_run", action="store_true", required=False, default=False, help="Dry run for testing. Will only translate 5 rows.")
    parser.add_argument("--hf_usage", action="store_true", required=False, default=False, help="Use vllm for translation.")

    # generation args
    parser.add_argument("--epsilon_cutoff", type=float, required=False, default=0.0)
    parser.add_argument("--num_return_sequences", type=int, required=False, default=1) # number of translations to generate for each source (used for reranking)!
    parser.add_argument("--temperature", type=float, required=False, default=0)
    parser.add_argument("--max_new_tokens", type=int, required=False, default=1024)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    return parser.parse_args()


def get_nllb_lang_code(lang):
    mapping = {
        "ar": "arb_Arab",
        "hi": "hin_Deva",
        "it": "ita_Latn",
        "es": "spa_Latn",
        "de": "deu_Latn",
        "pt": "por_Latn",
        "fr": "fra_Latn",
        "ru": "rus_Cyrl",
        "en": "eng_Latn",
    }
    return mapping[lang]

def main(
    args: argparse.Namespace
):
    
    # create output directory
    translation_model_name = args.model_name_or_path.replace("/", "--")
    output_dir = os.path.join(args.output_dir, f"{translation_model_name}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"translated-context_en_{args.tgt_lang}_test.txt")

    if os.path.exists(output_file):
        logger.info(f"File {output_file} already exists. Skipping generations.")
        exit()  


    # load dataset with contexts
    data = load_dataset("gsarti/mt_geneval", f"context_en_{args.tgt_lang}", split="test")
    logger.info(f"Loaded dataset {data}")
    logger.info(f"Len: {len(data)}")
    logger.info(f"Random row: {data[random.randint(0, len(data))]}")

    if args.dry_run:
        logger.info(f"Running dry run for 5 rows")
        data = data.select(range(5))

    data = data["context"]
    logger.info(f"Sample source to translate: {data[0]}")

    
    if args.hf_usage:
        # huggingface usage (for nllb models)
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,src_lang=get_nllb_lang_code("en"))
        generation_kwargs = {
        "do_sample": False,
        "num_beams": 5,
        "max_new_tokens": 256,
        "num_return_sequences": args.num_return_sequences,
        "forced_bos_token_id": tokenizer.convert_tokens_to_ids(get_nllb_lang_code(args.tgt_lang))
        }
        
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # logger.info(f"Using device: {device}")
        model = model.to(device)  # Move model to GPU

        # generate generations
        generations = []
        batch_size = args.batch_size
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        # Process data in batches with progress bar
        for i in tqdm(range(0, len(data), batch_size), total=total_batches, desc="Generating translations"):
            batch_samples = data[i:i + batch_size]
            
            # Tokenize the input
            inputs = tokenizer(batch_samples, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Generate with the specified parameters
            outputs = model.generate(
                **inputs,
                # max_length=args.max_new_tokens,
                # num_return_sequences=args.num_return_sequences,
                # temperature=args.temperature,
                # do_sample=True,  # Enable sampling for temperature to work
                # pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=tokenizer.eos_token_id,
                # forced_bos_token_id=tokenizer.convert_tokens_to_ids(get_nllb_lang_code(args.tgt_lang)),
                **generation_kwargs,
            )
            # Decode the generated sequences
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generations.extend(decoded_outputs)

        # write generations to file
        write_lines(output_file, generations)
    else:
        # vllm usage
        # initialize model and run generation
        model = LLM(
            model=args.model_name_or_path,
            seed=42,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            batch_size=args.batch_size,
        )
        
        s = SamplingParams(max_tokens=args.max_new_tokens, temperature=args.temperature, min_p=args.epsilon_cutoff, stop = ["\n", "\\n", "</s>"],)
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
            candidates.extend([l] * args.num_return_sequences)  
        model_output = model.generate(candidates, s, use_tqdm=True)
        generations = [output.outputs[0].text for output in model_output]

        # write generations to file
        write_lines(output_file, generations)

if __name__ == "__main__":
    args = parse_args()
    main(args)