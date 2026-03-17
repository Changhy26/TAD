import torch
import librosa
import numpy as np
import re, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, LogitsProcessor, Gemma3nForConditionalGeneration, AutoModelForSpeechSeq2Seq

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np



def plot_confusion_matrix(all_results, fig_name="confusion_matrix"):
    true_labels = [item['label'] for item in all_results]
    pred_labels = [item['yes_no'] for item in all_results]

    labels = ["yes", "no"]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels, normalize='true')

    os.makedirs('confusion_matrix', exist_ok=True)

    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'shrink': 0.8})

    ax.set_xlabel('Predicted labels', fontsize=14, fontweight='bold')
    ax.set_ylabel('True labels', fontsize=14, fontweight='bold')
    # ax.set_title("Confusion Matrix for Yes/No Predictions", fontsize=16, fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'confusion_matrix/{fig_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'confusion_matrix/{fig_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'confusion_matrix/{fig_name}.svg', dpi=300, bbox_inches='tight')

    plt.show()

    plt.close()

def plot_roc_curve(true_labels, pred_scores):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def build_yes_no_id_sets(tokenizer):
    yes_surfaces = ["yes", " yes", "Yes", " Yes", "是", " 有", "有"]
    no_surfaces = ["no", " no", "No", " No", "否", " 没有", "没有"]

    def encode_all(surfaces):
        all_ids = []
        for w in surfaces:
            ids = tokenizer.encode(w, add_special_tokens=False)
            all_ids.extend(ids)
        seen, kept = set(), []
        for i in all_ids:
            if i not in seen:
                kept.append(i)
                seen.add(i)
        return kept

    yes_ids = encode_all(yes_surfaces)
    no_ids = encode_all(no_surfaces)
    return yes_ids, no_ids


def _logsumexp_pool(logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """
    logits: (B, V)
    ids:    list[int] 
    return: (B,)
    """
    if not ids:
        return torch.full((logits.size(0),), -1e30, device=logits.device, dtype=logits.dtype)
    if len(ids) == 1:
        return logits[:, ids[0]]
    idx = torch.as_tensor(ids, device=logits.device, dtype=torch.long)
    sub = logits.index_select(dim=-1, index=idx)  # (B, |ids|)
    m = sub.max(dim=-1, keepdim=True).values  # (B,1)
    return m.squeeze(-1) + torch.log(
        torch.clamp(torch.exp(sub - m).sum(dim=-1), min=1e-8)
    )

def extract_yes_no(text):
    pattern = r"\b(yes|no)\b"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(0).lower()
    elif (
        "there is no sound" in text
        or "there is no sound of" in text
        or "there is no" in text
        or "is not" in text
    ):
        return "no"
    elif "does not contain" in text or "doesn't contain" in text:
        return "no"
    elif "contain" in text or "contains" in text:
        return "yes"
    elif "not" in text or "unable" in text or "can't" in text:
        return "no"
    else:
        return ""


def discriminative_metric(result):
    acc = 0
    precision = 0
    recall = 0
    f1 = 0
    yes_count = 0
    yes_true_positives = 0
    no_pred_count = 0
    true_positives = 0
    total_actual_no = 0
    total_actual_yes = 0
    not_answer_count = 0

    for res in result:
        if res["yes_no"] == "":
            not_answer_count += 1
            continue

        if res["yes_no"] == res["label"]:
            acc += 1

        if res["yes_no"] == "yes":
            yes_count += 1
            if res["label"] == "yes":
                yes_true_positives += 1

        if res["yes_no"] == "no":
            no_pred_count += 1
            if res["label"] == "no":
                true_positives += 1

        if res["label"] == "no":
            total_actual_no += 1
        if res["label"] == "yes":
            total_actual_yes += 1

    total = len(result)
    acc = acc / total if total > 0 else 0
    precision = true_positives / no_pred_count if no_pred_count > 0 else 0
    recall = true_positives / total_actual_no if total_actual_no > 0 else 0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    yes_ratio = yes_count / total if total > 0 else 0
    not_answer_rate = not_answer_count / total if total > 0 else 0

    print(
        f"Accuracy: {round(acc, 3)}, Precision: {round(precision, 3)}, "
        f"Recall: {round(recall, 3)}, F1: {round(f1, 3)}, "
        f"Yes_ratio: {round(yes_ratio, 3)}, Not answered ratio: {round(not_answer_rate, 3)}"
    )

    return (
        round(acc, 3),
        round(precision, 3),
        round(recall, 3),
        round(f1, 3),
        round(yes_ratio, 3),
        round(not_answer_rate, 3),
    )


def load_audio(qa_item, sampling_rate):
    """Load and process a single audio file"""
    try:
        audio_path = qa_item["path"]
        audio, sr = librosa.load(audio_path, sr=sampling_rate, mono=True)
        zero_audio = np.zeros_like(audio)  
        if sr != sampling_rate:
            print(
                f"Warning: Audio file {audio_path} was resampled from {sr} to {sampling_rate} Hz"
            )

        return {
            "audio": audio,
            "zero_audio": zero_audio,
            "qa_item": qa_item,
            "success": True,
        }
    except Exception as e:
        print(f"Error loading audio file {qa_item['path']}: {str(e)}")
        return {"success": False}


def cast_inputs_to_model_dtype(inputs, model):
    device = next(model.parameters()).device
    model_dtype = None
    for p in model.parameters():
        if p.is_floating_point():
            model_dtype = p.dtype
            break

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                inputs[k] = v.to(device=device, dtype=model_dtype)
            else:
                inputs[k] = v.to(device=device)
    return inputs

def load_model_and_processor(model_flag):
    if model_flag == 1:
        
        local_model_path = "./hf_cache/models/Qwen2-Audio-7B-Instruct"
        # processor（tokenizer + feature extractor）
        processor = AutoProcessor.from_pretrained(
            local_model_path,
            local_files_only=True  
        )
        # 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            local_model_path,
            local_files_only=True,      
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True      
        )
        model.tie_weights()

        # Initialize model and processor
        #processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        # Load single model
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        #model = Qwen2AudioForConditionalGeneration.from_pretrained(
        #    "Qwen/Qwen2-Audio-7B-Instruct",
        #    device_map=device,
        #    torch_dtype=torch.float16
        #)
        #model.tie_weights() 
    elif model_flag == 2:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "tsinghua-ee/SALMONN-7B",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("tsinghua-ee/SALMONN-7B", trust_remote_code=True)
        model.tie_weights()
    elif model_flag == 3:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_path = "./hf_cache/models/google/gemma-3n-E4B-it"
        processor = AutoProcessor.from_pretrained(local_path, local_files_only=True,)
        model = Gemma3nForConditionalGeneration.from_pretrained(local_path, device_map=device, torch_dtype=torch.float16, local_files_only=True,).eval()
        #model = Gemma3nForConditionalGeneration.from_pretrained("google/gemma-3n-E4B-it", device_map=device, torch_dtype=torch.float16).eval()
        
        model.tie_weights()
        model.forward = torch.compiler.disable(model.forward)
    else:
        raise ValueError("Invalid model flag.")

    return model, processor
