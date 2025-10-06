from comet import download_model, load_from_checkpoint
import torch
import transformers

import numpy as np
import pandas as pd
import datasets
from typing import List
from tqdm import tqdm
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError,
)
from openai import OpenAI
import re
from vllm import LLM, SamplingParams


# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
# Add the handlers to the logger
logger.addHandler(ch)

try:
    from metricx.metricx23 import models
except ImportError:
    logger.warning(
        "Could not import MetricX23 models. This is expected if you are not using MetricX23."
    )


id2lang = {
    # src
    "en": "English",
    # initial targets
    "it": "Italian",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    # extended targets
    "ar": "Arabic",
    "fr": "French",
    "hi": "Hindi",
    "ru": "Russian",
}


def extract_score(sent):
    sent = sent.lower()
    sent = re.sub(r"\*+", "", sent)

    patterns = [
        r"i would score this translation (\d+(\.\d+)?)",
        r"i would score this translation a (\d+(\.\d+)?)",
        r"i would score this translation an (\d+(\.\d+)?)",
        r"i would score this translation as (\d+(\.\d+)?)",
        r"i would score this translation around (\d+(\.\d+)?)",
        r"i would score it around (\d+(\.\d+)?)",
        r"i would give it a score of (\d+(\.\d+)?)",
        r"i would give this translation a score of (\d+(\.\d+)?)",
        r"i would give the translation a score of (\d+(\.\d+)?)",
        r"i'd give this translation a score of (\d+(\.\d+)?)",
        r"score: (\d+(\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, sent, re.IGNORECASE)
        if match:
            return float(match.group(1))

    # If no pattern matches, extract the first integer in the string
    match = re.search(r"\d+(\.\d+)?", sent)
    return float(match.group()) if match else None


def read_file(fname, unescape_newline=False):
    output = []
    with open(fname) as f:
        for line in f:
            output.append(line.strip())
    if unescape_newline:
        output = [l.replace("\\n", "\n") for l in output]
    return output


def write_lines(output_file, outputs):
    with open(output_file, "w") as f:
        for i, line in enumerate(outputs):
            if "\r" in line:
                print(
                    f"Detected carriage return in line {i + 1} (\\r). This may cause errors downstream. Escaping. This behaviour is the default; you can turn it off with escape_return_char."
                )
                line = line.replace("\r", "\\r")
            if "\n" in line:
                print(
                    f"Found new line in line {i + 1} (\\n). This may cause errors downstream. Escaping."
                )
                line = line.replace("\n", "\\n")
            f.write(f"{line}\n")


def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


class COMETScorer:
    def __init__(self, model_name_or_path: str) -> None:
        path = download_model(model_name_or_path)
        self.model = load_from_checkpoint(path)

    def score(self, texts, batch_size=8, gpus=torch.cuda.device_count()):
        return self.model.predict(texts, batch_size=batch_size, gpus=gpus)


class MetricXScorer:
    def __init__(self, model_name="google/metricx-23-qe-xl-v2p0"):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl")
        self.model = models.MT5ForRegression.from_pretrained(
            model_name, device_map="auto"
        )
        # self.model.cuda()
        self.model.eval()

    def _make_input(self, example):
        example["input"] = "candidate: " + example["mt"] + " source: " + example["src"]
        return example

    def _tokenize(self, example):
        return self.tokenizer(
            example["input"], max_length=1024, truncation=True, padding=False
        )

    def _remove_eos(self, example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]
        return example

    def _process_dataset(self, texts):
        ds = datasets.Dataset.from_pandas(pd.DataFrame(data=texts))
        ds = ds.map(self._make_input)
        ds = ds.map(self._tokenize)
        ds = ds.map(self._remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=self.model.device,
            output_all_columns=True,
        )
        return ds

    def score(self, texts, batch_size=1):
        ds = self._process_dataset(texts)

        training_args = transformers.TrainingArguments(
            per_device_eval_batch_size=batch_size,
            output_dir="./",
            dataloader_pin_memory=False,
        )
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
        )

        predictions, _, _ = trainer.predict(test_dataset=ds)
        return predictions


class GEMBAPromptBuilder:

    @staticmethod
    def apply_gemba_prompt(
        src_lang: str,
        tgt_lang: str,
        source: str,
        hypothesis: str,
        use_src_context: bool = False,
        src_ctx_type: str = "standard",
        context: str = None,
    ):

        if not use_src_context:
            t = f"""Score the following translation from {id2lang[src_lang]} to {id2lang[tgt_lang]} on a continuous scale from 0 to 100, where score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".
{id2lang[src_lang]} source: "{source}" 
{id2lang[tgt_lang]} translation: "{hypothesis}"
Score: """
        else:
            if src_ctx_type == "paraphrased":
                t = f"""Score the following translation from {id2lang[src_lang]} to {id2lang[tgt_lang]} on a continuous scale from 0 to 100, where score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".
You can use the preceding context to evaluate the translation of the source.
{id2lang[src_lang]} preceding context: "{context}"
{id2lang[src_lang]} source: "{source}" 
{id2lang[tgt_lang]} translation: "{hypothesis}"
Score: """
            else:
                t = f"""Score the following translation from {id2lang[src_lang]} to {id2lang[tgt_lang]} on a continuous scale from 0 to 100 given the preceding context, where score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".
{id2lang[src_lang]} preceding context: "{context}"
{id2lang[src_lang]} source: "{source}" 
{id2lang[tgt_lang]} translation: "{hypothesis}"
Score: """

        return t


class BaseModelScorer:
    def __init__(
        self,
        model_name_or_path: str,
        src_lang: str,
        tgt_lang: str,
        gemba_use_src_context: bool = False,
        gemba_src_ctx_type: str = "standard",
        gemba_num_shots: int = 0,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.gemba_use_src_context = gemba_use_src_context
        self.gemba_src_ctx_type = gemba_src_ctx_type
        self.gemba_num_shots = gemba_num_shots

        self.prompt_builder = GEMBAPromptBuilder()

    def apply_gemba_prompt(self, x: str, h: str, ctx: str = None):

        return self.prompt_builder.apply_gemba_prompt(
            self.src_lang,
            self.tgt_lang,
            x,
            h,
            self.gemba_use_src_context,
            self.gemba_src_ctx_type,
            ctx,
        )


class GembaModelScorer(BaseModelScorer):
    def __init__(
        self,
        model_name_or_path: str,
        src_lang: str,
        tgt_lang: str,
        gemba_use_src_context: bool = False,
        gemba_src_ctx_type: str = "standard",
        gemba_num_shots: int = 0,
    ) -> None:
        super().__init__(
            model_name_or_path,
            src_lang,
            tgt_lang,
            gemba_use_src_context,
            gemba_src_ctx_type,
            gemba_num_shots,
        )
        
        self.generator = LLM(
            model=model_name_or_path,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=0.8,
            max_model_len=1024,
            tensor_parallel_size=torch.cuda.device_count(),
        )

    def score(self, inputs, batch_size=8, extract_scores: bool = False):
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=256,
        )

        if not self.gemba_use_src_context:
            texts = [self.apply_gemba_prompt(i["src"], i["mt"]) for i in inputs]
        else:
            texts = [
                self.apply_gemba_prompt(i["src"], i["mt"], i["src_ctx"]) for i in inputs
            ]

        outputs = self.generator.generate(texts, sampling_params)
        text_outputs = [o.outputs[0].text for o in outputs]

        out = text_outputs
        if extract_scores:
            numeric_scores = [extract_score(o) for o in text_outputs]
            out = (out, numeric_scores)

        return out


class OpenAIModelScorer(BaseModelScorer):

    def __init__(
        self,
        model_name_or_path: str,
        src_lang: str,
        tgt_lang: str,
        gemba_use_src_context: bool = False,
        gemba_src_ctx_type: str = "standard",
        gemba_num_shots: int = 0,
    ):
        super().__init__(
            model_name_or_path,
            src_lang,
            tgt_lang,
            gemba_use_src_context,
            gemba_src_ctx_type,
            gemba_num_shots,
        )

        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            logger.warning(
                "Could not import python-dotenv. Set environment variables manually."
            )

        self.client = OpenAI()

    def score(
        self,
        inputs: List[str],
        show_progress_bar: bool = True,
    ):
        """Generate GPT4 completions using local images and prompts."""

        if not self.gemba_use_src_context:
            texts = [self.apply_gemba_prompt(i["src"], i["mt"]) for i in inputs]
        else:
            texts = [
                self.apply_gemba_prompt(i["src"], i["mt"], i["src_ctx"]) for i in inputs
            ]

        logger.info(f"Sample input: {texts[0]}")

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(payload):
            return self.client.chat.completions.create(**payload)

        completions = list()
        for idx, prompt in tqdm(
            enumerate(texts),
            desc="Item",
            total=len(texts),
            disable=not show_progress_bar,
        ):
            payload = {
                "model": self.model_name_or_path,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": 256,
                "temperature": 0.0,
            }

            try:
                chat_response = completion_with_backoff(payload)
                response = chat_response.choices[0].message.content
            except RetryError:
                logger.warning(f"Retrying with OPENAI API failed.")
                logger.warning(f"Failing row {idx}, prompt: {prompt}")
                response = "FAILED"

            completions.append(response)

        return completions


def apply_template_tower(x, src_lang, tgt_lang):
    return f"Translate the following text from {id2lang.get(src_lang)} into {id2lang.get(tgt_lang)}.\n{id2lang.get(src_lang)}: {x}.\n{id2lang.get(tgt_lang)}:"


# https://github.com/amazon-science/machine-translation-gender-eval/blob/main/accuracy_metric.py
def get_words(line):
    """
    Helper function to get the set of words in a line.

    :param line: Line from which to get the words.
    :return: Set of words in the line.
    """
    return set(line.strip().split())


def get_trg_correct_incorrect(trg_line, orig_ref, ctf_ref):
    """
    Compute overlap between references and translation
    We first get unique words in each of the references w.r.t each other then we compute their overlap with target
    """
    # get words for each segment
    trg_words, orig_words, ctf_words = (
        get_words(trg_line),
        get_words(orig_ref),
        get_words(ctf_ref),
    )
    # get unique words in each of the references
    orig_unique = orig_words - ctf_words
    ctf_unique = ctf_words - orig_words
    # now check the words in the target sentence for overlap with incorrect unique words
    trg_correct = trg_words & orig_unique
    trg_incorrect = trg_words & ctf_unique
    return trg_correct, trg_incorrect


def gender_decision(trg_line, orig_ref, ctf_ref):
    """
    Check if gender of a sentence is correct based on corresponding correct and incorrect references.
    Algorithm: We make decision based on whether hyp overlaps with original ref and counterfactual ref

    :param trg_line: Sentence from translation output for which to check gender.
    :param orig_ref: Original (Correct) reference.
    :param ctf_ref: Counterfactual reference.
    :return: a list of decision, overlap(hyp, original ref), overlap(hyp, counterfactual ref)
    """
    trg_correct, trg_incorrect = get_trg_correct_incorrect(trg_line, orig_ref, ctf_ref)

    if trg_incorrect:
        decision = "Incorrect"
    else:
        decision = "Correct"

    return [decision, trg_correct, trg_incorrect]
