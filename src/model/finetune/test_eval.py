import argparse
import time
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
from peft import PeftModel
from dataset import DatasetLoader
from train import HF_TOKEN
from transformers import BitsAndBytesConfig
import decord
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor
import difflib
import statistics
from typing import List, Dict
from collections import Counter

# Constants matching train.py
NUM_FRAMES = 8

def sample_frames(video_path: str, num_frames: int = NUM_FRAMES):
    """
    Sample frames from a video using Decord.
    """
    if video_path is None or not Path(video_path).exists():
        raise ValueError(f"Invalid video path: {video_path}")

    vr = decord.VideoReader(video_path)
    total = len(vr)
    indices = torch.linspace(0, total - 1, num_frames).long()
    frames = vr.get_batch(indices).asnumpy()

    frames = torch.tensor(frames).permute(0, 3, 1, 2)  # Convert to (num_frames, C, H, W)
    return frames

class VLMEvaluator:
    def __init__(self, checkpoint_path: str = None, output_file: str = "eval_results.json", verbose: bool = True):
        self.verbose = verbose
        self.output_file = output_file
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_loader = self.load_model()
        self.model = model_loader["model"]
        self.processor = model_loader["processor"]
        self.processor.video_processor.size = {"height": 448, "width": 448}

        # Load LoRA Adapter if provided
        if self.checkpoint_path:
            if self.verbose:
                print(f"Loading LoRA adapter from {self.checkpoint_path}...")
            self.model = PeftModel.from_pretrained(self.model, self.checkpoint_path)
            self.model.eval()
            if self.verbose:
                print("\n\n✅ LoRA adapter loaded.")

        else:
            print(f"\n\n 🚨 No LoRA checkpoint provided, using base model.")
            self.model = model_loader["model"]
        
    def load_model(self):

        """
        Load the base model and attach the LoRA adapter for evaluation.

        This loads the processor (for text+video preprocessing) and the base model
        using 4-bit quantization where supported, then applies the PEFT adapter.
        """
        MODEL_ID = "OpenGVLab/InternVL3_5-2B-hf"
        print(f"\n Loading base model: {MODEL_ID}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )

        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
        processor.video_processor.size = {"height": 448, "width": 448}

        # Load Base Model
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
        )

        print("\n\n✅ Model loaded successfully.  Device:", self.device)


        return {
            "model": model,
            "processor": processor
        }
        pass
    def generate_response(self, video_path: str, prompt: str):
        # Prepare inputs
        start_time = time.time()
        video_tensor = sample_frames(video_path, NUM_FRAMES)
        
        if video_tensor is None:
            print(f"⚠️ Failed to sample frames for {video_path}")
            return ""
        
        prompt = "How are you?"

        print(f"\nSampled video frames shape: {video_tensor.shape} from {video_path}")

        # CRITICAL FIX: Convert (T, C, H, W) tensor back to (T, H, W, C) numpy array for processor
        # The processor expects a list of numpy arrays, usually uint8 [0-255]
        if isinstance(video_tensor, torch.Tensor):
            # Permute to (T, H, W, C) and move to CPU numpy
            video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
        else:
            video_np = video_tensor

         
        conversation = [
            {
                "role": "user",
                "content": f"<video>\n{prompt}"
            }
        ]
        
        # Apply chat template
        prompt_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # DEBUG: Print the exact prompt being sent to the model
        print(f"\n[DEBUG] Prompt Text: {prompt_text[-100:]!r}") # Check the end of the prompt

        # Process inputs
        inputs = self.processor(
            text=[prompt_text],
            videos=[video_np],
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = inputs.to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

        # Generate with repetition penalty to break the newline loop
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=64,  # Reduce max tokens since glosses are short
                min_new_tokens=2,
                do_sample=False,
                repetition_penalty=1.2,  # CRITICAL: Penalize repeating newlines
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            
            # Decode
            input_len = inputs['input_ids'].shape[-1]
            generated_tokens = output[0, input_len:]
            decoded_output = self.processor.decode(generated_tokens, skip_special_tokens=True)
            
            # Strip whitespace (newlines)
            result = decoded_output.strip()
            
            # DEBUG: Show what we got
            print(f"[DEBUG] Raw Result: {result!r}")

            stop_time = time.time()
            elapsed_time = stop_time - start_time
            print(f"\n📝 Generated VLM Model Response: '{result}' in {elapsed_time:.2f} secs")
        
        return result

    def evaluate(self, limit: int = None):
        
         # --- SANITY CHECK START ---
        print("\n🧪 Running Text-Only Sanity Check...")
        sanity_prompt = "Hello, who are you?"
        sanity_conv = [{"role": "user", "content": sanity_prompt}]
        sanity_text = self.processor.apply_chat_template(sanity_conv, tokenize=False, add_generation_prompt=True)
        sanity_inputs = self.processor(text=[sanity_text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            sanity_out = self.model.generate(
                **sanity_inputs, 
                max_new_tokens=20, 
                do_sample=False
            )
            sanity_res = self.processor.decode(sanity_out[0], skip_special_tokens=False)
            print(f"🧪 Sanity Result (Raw): {sanity_res}")
        # --- SANITY CHECK END ---


        # Load Dataset
        dataset_loader = DatasetLoader(verbose=self.verbose)
        # Use test data if available, otherwise fallback to validation data
        eval_data = dataset_loader.test_data if dataset_loader.test_data else dataset_loader.val_data
        
        results: List[dict] = []
        per_sample_metrics: List[dict] = []

       

        # Clear output file if it exists
        if os.path.exists(self.output_file):
            os.remove(self.output_file)


        if limit:
            eval_data = eval_data[:limit]


        print(f"Starting evaluation on {len(eval_data)} samples...")
        for item in tqdm(eval_data):
            video_path = "./data_engineering/"+item["video_path"]
            prompt = item["prompt"]
            gloss = item["gloss"]

            try:
                generated_gloss = self.generate_response(video_path, prompt)

                result = {
                    "video_path": video_path,
                    "prompt": prompt,
                    "gloss": gloss,
                    "generated_gloss": generated_gloss,
                }
                results.append(result)

                # Compute lightweight, dependency-free metrics
                em = _exact_match(gloss, generated_gloss)
                lev = _normalized_edit_ratio(gloss, generated_gloss)
                f1 = _token_f1(gloss, generated_gloss)
                bleu = _bleu_score(gloss, generated_gloss)
                rouge_l = _rouge_l(gloss, generated_gloss)

                sample_metrics = {
                    "video_path": video_path,
                    "exact_match": em,
                    "levenshtein_ratio": lev,
                    "token_f1": f1,
                    "bleu": bleu,
                    "rouge_l_precision": rouge_l["precision"],
                    "rouge_l_recall": rouge_l["recall"],
                    "rouge_l_f1": rouge_l["f1"],
                }
                per_sample_metrics.append(sample_metrics)

                # # Save per-sample result + metrics incrementally to JSONL
                # out_obj = {**result, "metrics": sample_metrics}
                # with open(self.output_file, "a", encoding="utf-8") as f:
                #     f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error processing {video_path}: {e}")

        # Aggregate metrics
        if per_sample_metrics:
            agg = _aggregate_metrics(per_sample_metrics)
        else:
            agg = {}

        # Attach per-sample metrics to each result (if available)
        if per_sample_metrics and len(per_sample_metrics) == len(results):
            for i, r in enumerate(results):
                r["metrics"] = per_sample_metrics[i]
        else:
            # If lengths mismatch, map metrics by video_path when possible
            metrics_by_video = {m.get("video_path"): m for m in per_sample_metrics}
            for r in results:
                r["metrics"] = metrics_by_video.get(r.get("video_path"), {})

        # Write full results array to a single JSON file
        out_path = Path(self.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Write aggregated metrics to separate JSON
        metrics_path = out_path.with_suffix('.metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as mf:
            json.dump(agg, mf, indent=2, ensure_ascii=False)

        print(f"Evaluation complete. Results saved to {out_path}")
        print(f"Aggregated metrics saved to {metrics_path}")


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    # simple normalization: lower, strip, collapse whitespace
    return " ".join(s.lower().strip().split())


def _exact_match(a: str, b: str) -> int:
    return int(_normalize_text(a) == _normalize_text(b))


def _normalized_edit_ratio(a: str, b: str) -> float:
    """Return a similarity ratio [0..1] based on SequenceMatcher."""
    a_n = _normalize_text(a)
    b_n = _normalize_text(b)
    if not a_n and not b_n:
        return 1.0
    return float(difflib.SequenceMatcher(None, a_n, b_n).ratio())


def _token_f1(a: str, b: str) -> float:
    """Compute token-level precision/recall/F1 (like QA overlap F1)."""
    a_tokens = _normalize_text(a).split()
    b_tokens = _normalize_text(b).split()
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    common = {}
    for t in a_tokens:
        common[t] = common.get(t, 0) + 1
    match = 0
    for t in b_tokens:
        if common.get(t, 0) > 0:
            match += 1
            common[t] -= 1
    precision = match / len(b_tokens)
    recall = match / len(a_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _bleu_score(reference: str, candidate: str, max_n: int = 4) -> float:
    """
    Compute BLEU score (simplified, sentence-level).
    Standard metric for evaluating text generation quality in VLMs.
    """
    ref_tokens = _normalize_text(reference).split()
    cand_tokens = _normalize_text(candidate).split()
    
    if not cand_tokens:
        return 0.0
    
    # Compute n-gram precisions
    precisions = []
    for n in range(1, min(max_n + 1, len(cand_tokens) + 1)):
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])
        cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)])
        
        overlap = sum((ref_ngrams & cand_ngrams).values())
        total = sum(cand_ngrams.values())
        
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(overlap / total)
    
    if not precisions or all(p == 0 for p in precisions):
        return 0.0
    
    # Geometric mean of precisions
    from math import exp, log
    geo_mean = exp(sum(log(p) if p > 0 else float('-inf') for p in precisions) / len(precisions))
    
    if geo_mean == 0.0 or geo_mean == float('-inf'):
        return 0.0
    
    # Brevity penalty
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    if cand_len >= ref_len:
        bp = 1.0
    else:
        bp = exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    
    return bp * geo_mean


def _rouge_l(reference: str, candidate: str) -> Dict[str, float]:
    """
    Compute ROUGE-L score (Longest Common Subsequence).
    Widely used for evaluating VLM and summarization quality.
    """
    ref_tokens = _normalize_text(reference).split()
    cand_tokens = _normalize_text(candidate).split()
    
    if not ref_tokens or not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Compute LCS length using dynamic programming
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[m][n]
    
    precision = lcs_len / n if n > 0 else 0.0
    recall = lcs_len / m if m > 0 else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}


def _aggregate_metrics(per_sample_metrics: List[dict]) -> dict:
    ems = [m["exact_match"] for m in per_sample_metrics]
    levs = [m["levenshtein_ratio"] for m in per_sample_metrics]
    f1s = [m["token_f1"] for m in per_sample_metrics]
    bleus = [m["bleu"] for m in per_sample_metrics]
    rouge_f1s = [m["rouge_l_f1"] for m in per_sample_metrics]

    return {
        "num_samples": len(per_sample_metrics),
        "exact_match": {
            "mean": statistics.mean(ems),
            "median": statistics.median(ems),
        },
        "levenshtein_ratio": {
            "mean": statistics.mean(levs),
            "median": statistics.median(levs),
        },
        "token_f1": {
            "mean": statistics.mean(f1s),
            "median": statistics.median(f1s),
        },
        "bleu": {
            "mean": statistics.mean(bleus),
            "median": statistics.median(bleus),
        },
        "rouge_l_f1": {
            "mean": statistics.mean(rouge_f1s),
            "median": statistics.median(rouge_f1s),
        },
    }

if __name__ == "__main__":
    
    checkpoint = "./vlm_finetuned/checkpoint-120"  # Replace with actual checkpoint path if needed
    
    evaluator = VLMEvaluator(checkpoint_path=checkpoint, output_file="evaluation_results/eval_results.json")
    evaluator.evaluate(limit=3)