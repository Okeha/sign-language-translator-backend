import os
import torch
import json
import yaml
import time
import statistics
import difflib
from tqdm import tqdm
from typing import List, Dict
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
import decord
from dataset import DatasetLoader
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Load configuration
with open("../../params/vlm.yml", "r") as f:
    params = yaml.safe_load(f)

HF_TOKEN = os.getenv('HF_TOKEN')
NUM_FRAMES = 8
MODEL_ID = params["model"]
# Default to the final model or a specific checkpoint
def get_latest_checkpoint(base_dir="./vlm_finetuned"):
    """
    Return the path to the latest numeric checkpoint directory under `base_dir`.

    Args:
        base_dir (str): Directory containing checkpoint-* folders.

    Returns:
        str|None: Path to the latest checkpoint or None if none exist.
    """
    if not os.path.exists(base_dir):
        return None
    checkpoints = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    # Sort by number
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(base_dir, checkpoints[-1])

ADAPTER_PATH = get_latest_checkpoint() or "./vlm_finetuned_final"


# ==================== METRIC HELPERS ====================

def _normalize_text(s: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    if s is None:
        return ""
    return " ".join(s.lower().strip().split())


def _exact_match(a: str, b: str) -> int:
    """Return 1 if normalized texts match exactly, 0 otherwise."""
    return int(_normalize_text(a) == _normalize_text(b))


def _normalized_edit_ratio(a: str, b: str) -> float:
    """
    Return a similarity ratio [0..1] based on Levenshtein distance using SequenceMatcher.
    Standard metric for evaluating text similarity in VLMs.
    """
    a_n = _normalize_text(a)
    b_n = _normalize_text(b)
    if not a_n and not b_n:
        return 1.0
    return float(difflib.SequenceMatcher(None, a_n, b_n).ratio())


def _token_f1(a: str, b: str) -> float:
    """
    Compute token-level precision/recall/F1 (like QA overlap F1).
    Standard metric for evaluating token overlap in VLM predictions.
    """
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
    """Aggregate per-sample metrics into mean/median statistics."""
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

# ==================================================================

def sample_frames(video_path: str, num_frames: int = NUM_FRAMES):
    """
    Sample frames from a video using Decord.
    """
    print("\nSampling frames from video:", video_path)
    if video_path is None or not Path(video_path).exists():
        raise ValueError(f"Invalid video path: {video_path}")

    vr = decord.VideoReader(video_path)
    total = len(vr)
    indices = torch.linspace(0, total - 1, num_frames).long()
    frames = vr.get_batch(indices).asnumpy()

    # frames = torch.tensor(frames).permute(0, 3, 1, 2)  # Convert to (num_frames, C, H, W)
    return frames

class VLMEvaluator:
    def __init__(self, adapter_path=None):
        """
        Initialize the evaluator.

        Args:
            adapter_path (str, optional): Path to the PEFT adapter/checkpoint. If None,
                uses latest found checkpoint or `vlm_finetuned_final`.

        Side effects:
            Loads the model processor and LoRA adapter via `load_model`.
        """
        self.adapter_path = adapter_path or ADAPTER_PATH
        if not self.adapter_path or not os.path.exists(self.adapter_path):
            # raise ValueError(f"Adapter path not found: {self.adapter_path}")
            # print(f"Adapter path not found: {self.adapter_path}")
            pass

            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        print(f"\n\n 🚀 Initializing Evaluator with adapter: {self.adapter_path}")
        self.load_model()

    def load_model(self):
        """
        Load the base model and attach the LoRA adapter for evaluation.

        This loads the processor (for text+video preprocessing) and the base model
        using 4-bit quantization where supported, then applies the PEFT adapter.
        """
        print(f"\nLoading base model: {MODEL_ID}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )

        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
        self.processor.video_processor.size = {"height": 448, "width": 448}

        # Load Base Model
        base_model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
        )

        
        # Load LoRA Adapter
        print(f"\nLoading LoRA adapter from {self.adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.to(self.device)

        # CORRECT CHECK: Verify LoRA modules exist (not trainability)
        lora_modules = [name for name, module in self.model.named_modules() if 'lora' in name.lower()]
        if lora_modules:
            print(f"✅ LoRA adapter loaded successfully ({len(lora_modules)} LoRA modules found)")
            print(f"   Example modules: {lora_modules[:3]}")
        else:
            print("⚠️ WARNING: No LoRA modules found! Adapter may not be applied.")
        
        # Optional: Print parameter stats (trainable will be 0 for inference, this is normal)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🔢 Parameters: Trainable={trainable_params:,} / Total={total_params:,} (Trainable=0 is normal for eval)")
        
        self.model.eval()
        
        print("\n\n✅ Model loaded successfully.  Device:", self.device)

    # def sample_frames(self, video_path, num_frames=NUM_FRAMES):
    #     """
    #     Sample `num_frames` frames from `video_path` for evaluation.

    #     Args:
    #         video_path (str): Path to the video (may be relative to dataset entries).
    #         num_frames (int): Number of frames to sample.

    #     Returns:
    #         numpy.ndarray|None: Frames array (T,H,W,C) or None on error.
    #     """
    #     try:
    #         # Handle relative paths from the dataset
    #         if not os.path.exists(video_path):
    #             # Try prepending data_engineering if not found
    #             alt_path = os.path.join("data_engineering", video_path)
    #             if os.path.exists(alt_path):
    #                 video_path = alt_path
    #             else:
    #                 # Try absolute path construction if needed
    #                 pass

    #         vr = decord.VideoReader(video_path)
    #         total = len(vr)
    #         indices = torch.linspace(0, total - 1, num_frames).long()
    #         frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)

    #         frames = torch.tensor(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
    #         return frames
        
    #     except Exception as e:
    #         print(f"\n⚠️ Error sampling frames from {video_path}: {e}")
    #         return None

    def generate_response(self, video_path, prompt):
        """
        Generate a response from the model given a video and prompt.

        Args:
            video_path (str): Path to the video file.
            prompt (str): Text prompt to accompany the video.
        
        Returns:
            str: Generated text response
        """
        start_time = time.time()
        video_tensor = sample_frames(video_path, num_frames=NUM_FRAMES)
        
        # if video_tensor is None:
        #     print(f"⚠️ Failed to sample frames for {video_path}")
        #     return ""
        
        # print(f"\nSampled video frames shape: {video_tensor.shape} from {video_path}")
        
        # # CRITICAL FIX: Convert (T, C, H, W) tensor to (T, H, W, C) numpy array
        # if isinstance(video_tensor, torch.Tensor):
        #     video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
        # else:
        #     video_np = video_tensor
        
        # Create text-only conversation (video handled separately)
        conversation = [
            {
                "role": "user",
                "content": f"<video>\n{prompt}"
            }
        ]
        
        print(video_path)
        # # Apply chat template to text only
        prompt_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # # FIX: Pass numpy array to processor, not tensor
        inputs = self.processor(
            text=[prompt_text],
            videos=[video_tensor],  # Pass the numpy array here
            return_tensors="pt",
            padding=True
        ).to(self.device, dtype = torch.float16)
        
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                min_new_tokens=1,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

            # 🕵️ DEBUG: Print Raw Tokens
            generated_ids = output[0, inputs['input_ids'].shape[-1]:]
            print(f"🔢 RAW GENERATED TOKENS: {generated_ids.tolist()}")


            # Decode only the generated part (skip input tokens)
            decoded_output = self.processor.decode(
                output[0, inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            )
            

            result = decoded_output.strip()
            elapsed_time = time.time() - start_time
            print(f"\n📝 Generated VLM Model Response: '{result}' in {elapsed_time:.2f} secs")
        
        return result
    
    def evaluate(self, test_data=None, limit=None):
        """
        Run evaluation on `test_data` (or dataset loader's test set) and compute Top-1/Top-5 metrics.

        Args:
            test_data (list, optional): List of dataset samples to evaluate. If None, loads the default test set.
            limit (int, optional): If set, evaluate only the first `limit` samples.

        Returns:
            dict: Dictionary containing Top-1 accuracy, Top-5 accuracy, and aggregated text metrics
        """
        if test_data is None:
            print("Loading test dataset...")
            loader = DatasetLoader(verbose=False)
            test_data = loader.test_data

        if limit:
            test_data = test_data[:limit]

        print(f"Starting evaluation on {len(test_data)} samples...")
        
        predictions = []
        ground_truths = []
        results = []
        per_sample_metrics: List[dict] = []

        for i, sample in tqdm(enumerate(test_data), total=len(test_data)):
            video_path =  "./data_engineering/"+ sample['video_path']
            ground_truth = sample['gloss'].lower().strip()
            prompt = sample['prompt']

            print(prompt)

            generated_text = self.generate_response(video_path, prompt)
            generated_texts = [generated_text] if generated_text else []

            # Clean up predictions
            top_preds = [t.strip().lower() for t in generated_texts if t]
            # Remove duplicates while preserving order
            seen = set()
            top_preds_unique = []
            for x in top_preds:
                if x not in seen:
                    top_preds_unique.append(x)
                    seen.add(x)
            
            # Pad if fewer than 5 unique predictions
            while len(top_preds_unique) < 5:
                top_preds_unique.append("")

            top1_pred = top_preds_unique[0] if top_preds_unique else ""
            
            predictions.append(top1_pred)
            ground_truths.append(ground_truth)
            
            is_correct_top1 = top1_pred == ground_truth
            is_correct_top5 = ground_truth in top_preds_unique[:5]
            
            # Compute text-level metrics (using Top-1 prediction)
            em = _exact_match(ground_truth, top1_pred)
            lev = _normalized_edit_ratio(ground_truth, top1_pred)
            f1 = _token_f1(ground_truth, top1_pred)
            bleu = _bleu_score(ground_truth, top1_pred)
            rouge_l = _rouge_l(ground_truth, top1_pred)
            
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
            
            results.append({
                "video_path": video_path,
                "ground_truth": ground_truth,
                "prediction_top1": top1_pred,
                "prediction_top5": top_preds_unique[:5],
                "correct_top1": is_correct_top1,
                "correct_top5": is_correct_top5,
                "metrics": sample_metrics
            })

            if i % 10 == 0:
                print(f"Step {i}: GT='{ground_truth}' | Pred='{top1_pred}' | Top5={is_correct_top5}")

        # Calculate Top-1/Top-5 Accuracy Metrics
        accuracy_top1 = accuracy_score(ground_truths, predictions)
        accuracy_top5 = sum([r['correct_top5'] for r in results]) / len(results) if results else 0.0
        
        # Aggregate text-level metrics
        if per_sample_metrics:
            text_metrics_agg = _aggregate_metrics(per_sample_metrics)
        else:
            text_metrics_agg = {}
        
        print(f"\n📊 Evaluation Results:")
        print(f"✅ Top-1 Accuracy: {accuracy_top1:.4f}")
        print(f"✅ Top-5 Accuracy: {accuracy_top5:.4f}")
        print(f"\n📝 Text-Level Metrics (Mean):")
        if text_metrics_agg:
            print(f"  - Exact Match: {text_metrics_agg['exact_match']['mean']:.4f}")
            print(f"  - Levenshtein Ratio: {text_metrics_agg['levenshtein_ratio']['mean']:.4f}")
            print(f"  - Token F1: {text_metrics_agg['token_f1']['mean']:.4f}")
            print(f"  - BLEU: {text_metrics_agg['bleu']['mean']:.4f}")
            print(f"  - ROUGE-L F1: {text_metrics_agg['rouge_l_f1']['mean']:.4f}")

        # Detailed Report
        print("\n📝 Classification Report (Top-1):")
        # Get unique labels from both lists to handle cases where some classes might not be predicted
        unique_labels = sorted(list(set(ground_truths + predictions)))
        print(classification_report(ground_truths, predictions, zero_division=0))

        # Save detailed results to JSON (not JSONL)
        output_file = "evaluation_results.json"
        output_data = {
            "metrics": {
                "accuracy_top1": accuracy_top1,
                "accuracy_top5": accuracy_top5,
            },
            "text_metrics_aggregated": text_metrics_agg,
            "details": results
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Save aggregated metrics separately
        metrics_file = "evaluation_metrics_summary.json"
        summary_data = {
            "num_samples": len(results),
            "accuracy_metrics": {
                "top1_accuracy": accuracy_top1,
                "top5_accuracy": accuracy_top5,
            },
            "text_metrics": text_metrics_agg
        }
        
        with open(metrics_file, "w", encoding="utf-8") as mf:
            json.dump(summary_data, mf, indent=2, ensure_ascii=False)
        
        print(f"💾 Detailed results saved to {output_file}")
        print(f"💾 Aggregated metrics saved to {metrics_file}")
        
        return {
            "accuracy_top1": accuracy_top1,
            "accuracy_top5": accuracy_top5,
            "text_metrics": text_metrics_agg
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    checkpoint = "./vlm_finetuned_final"

    evaluator = VLMEvaluator(adapter_path=checkpoint)
    evaluator.evaluate(limit=3)
