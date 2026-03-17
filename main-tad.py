import json
import librosa
import torch
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, LogitsProcessor, Gemma3nForConditionalGeneration
from transformers import AutoModelForSpeechSeq2Seq
from tqdm import tqdm
import re
import numpy as np
from datetime import datetime
import gc
import os

from utils import (
    build_yes_no_id_sets,
    _logsumexp_pool,
    extract_yes_no,
    discriminative_metric,
    load_audio,
    cast_inputs_to_model_dtype,
    load_model_and_processor,
    plot_confusion_matrix
)


'''
Logits processor is the key component for AAD, it 
modifies the logits of the next token to be generated
during the generation process
'''

class AudioLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        model,
        embeds_without_audio,
        atts_without_audio,
        alpha: float = 0.5,
        yes_token_ids=None,
        no_token_ids=None,
        tau_margin: float = None,
        penalty_value: float = 0.0,
    ):
        self.model = model
        self.alpha = alpha
        self.embeds_without_audio = embeds_without_audio
        self.atts_without_audio = atts_without_audio
        self.first_call = True
        self.step = 0

        self.yes_ids = list(yes_token_ids or [])
        self.no_ids  = list(no_token_ids or [])
        self.tau_margin = tau_margin
        self.penalty_value = penalty_value

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            # no audio embeddings/mask ——
            if not self.first_call:
                new_token = input_ids[:, -1:]  # (B,1)
                new_emb = self.model.get_input_embeddings()(new_token)
                self.embeds_without_audio = torch.cat([self.embeds_without_audio, new_emb], dim=1)
                new_attn = torch.ones_like(new_token, dtype=self.atts_without_audio.dtype)
                self.atts_without_audio = torch.cat([self.atts_without_audio, new_attn], dim=1)
            else:
                self.first_call = False
                
            outputs_wo = self.model(
                inputs_embeds=self.embeds_without_audio,
                attention_mask=self.atts_without_audio
            )
            logits_wo = outputs_wo.logits[:, -1, :]  # (B,V)

            # contrastive decoding
            modified_logits = (1.0 + self.alpha) * scores - self.alpha * logits_wo

            # first-step gating
            if (self.step == 0 and self.tau_margin is not None and self.penalty_value > 0.0
                and self.yes_ids and self.no_ids):

                # logsumexp_pool for audio and salient
                ay = _logsumexp_pool(scores,    self.yes_ids)   # (B,)
                an = _logsumexp_pool(scores,    self.no_ids)    # (B,)
                by = _logsumexp_pool(logits_wo, self.yes_ids)   # (B,)
                bn = _logsumexp_pool(logits_wo, self.no_ids)    # (B,)

                # margin for yes
                delta_margin = (ay - an) - (by - bn)            # (B,)

                # Penalize if audio evidence is insufficient (below threshold).
                trigger = (delta_margin < self.tau_margin)
                if trigger.any():
                    idx_yes = torch.as_tensor(self.yes_ids, device=modified_logits.device, dtype=torch.long)
                    rows = torch.nonzero(trigger, as_tuple=False).squeeze(-1).tolist()
                    for b in rows:
                        modified_logits[b, idx_yes] -= self.penalty_value

            self.step += 1
            return modified_logits


def process_batch(batch_data, model, processor):
    """Process a batch of data with the model"""
    
    sampling_rate = processor.feature_extractor.sampling_rate
    batch_results = []
    
    try:
        # Unpack the batch data
        valid_batch, audios, valid_texts, zero_audios = batch_data
        
        # Prepare inputs 
        inputs = processor(text=valid_texts, audio=audios, return_tensors="pt", padding=True, sampling_rate=sampling_rate)
        if model_flag == 3:
            inputs = cast_inputs_to_model_dtype(inputs, model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Only prepare no-audio inputs if using logits processor  
        if use_logits_processor:
            inputs_without_audio = processor(text=valid_texts, audio=None, return_tensors="pt", padding=True, sampling_rate=sampling_rate)
            # you can also put zero_audios here
            # inputs_without_audio = processor(text=valid_texts, audio=zero_audios, return_tensors="pt", padding=True, sampling_rate=sampling_rate)
            inputs_without_audio = {k: v.to(device) for k, v in inputs_without_audio.items()} 
            
            # Get input embeddings directly from the embedding layer
            input_embeddings = model.get_input_embeddings()(inputs_without_audio["input_ids"])

            tok = processor.tokenizer
            yes_ids, no_ids = build_yes_no_id_sets(tok)

            logits_processor = AudioLogitsProcessor(
                model,
                input_embeddings,
                inputs_without_audio["attention_mask"],
                alpha=alpha,
                yes_token_ids=yes_ids,
                no_token_ids=no_ids,
                tau_margin=0.2,  # 0.0~0.5
                penalty_value=2.5  # 2~4
            )


            logits_processor_list = [logits_processor] 

        else:
            logits_processor_list = []

        # ====== ROC score: P(no) from first-step logits ======
        # yes/no token id set
        yes_ids, no_ids = build_yes_no_id_sets(processor.tokenizer)

        with torch.no_grad():
            # with-audio first-step next-token logits
            out_with = model(**inputs)
            logits_with = out_with.logits[:, -1, :]  # (B, V)

            if use_logits_processor:
                # without-audio first-step logits
                out_wo = model(**inputs_without_audio)
                logits_wo = out_wo.logits[:, -1, :]  # (B, V)

                # AAD first-step logits
                logits_used = (1 + alpha) * logits_with - alpha * logits_wo
                logits_pre = logits_used.detach().clone()  # pre-penalty logits (copy)
            else:
                # Default
                logits_used = logits_with

            # TAD step-0 gating (for ROC)
            if use_logits_processor:
                # Reuse the same tau/penalty as the logits_processor used in generation
                tau_margin = getattr(logits_processor_list[0], 'tau_margin', None) if logits_processor_list else None
                penalty_value = getattr(logits_processor_list[0], 'penalty_value', 0.0) if logits_processor_list else 0.0

                if (tau_margin is not None and penalty_value > 0.0 and yes_ids and no_ids):
                    ay = _logsumexp_pool(logits_with, yes_ids)
                    an = _logsumexp_pool(logits_with, no_ids)
                    by = _logsumexp_pool(logits_wo, yes_ids)
                    bn = _logsumexp_pool(logits_wo, no_ids)

                    delta = (ay - an) - (by - bn)
                    mask = delta < tau_margin  # (B,)

                    if mask.any():
                        row_idx = mask.nonzero(as_tuple=True)[0]
                        col_idx = torch.as_tensor(yes_ids, device=logits_used.device, dtype=torch.long)
                        logits_used = logits_used.clone()
                        logits_used[row_idx[:, None], col_idx[None, :]] -= penalty_value            
                logits_post = logits_used.detach().clone()  # post-penalty logits (copy)

            # yes/no pooling 
            L_yes = _logsumexp_pool(logits_used, yes_ids)  # (B,)
            L_no = _logsumexp_pool(logits_used, no_ids)  # (B,)

            # softmax for P(no)
            # P(no)=exp(L_no)/(exp(L_no)+exp(L_yes))
            p_no = torch.sigmoid(L_no - L_yes).detach().cpu().numpy()  # (B,)
        # =====================================================

            # Evidence terms for analysis (first-step)
            if use_logits_processor:
                ay = _logsumexp_pool(logits_with, yes_ids)
                an = _logsumexp_pool(logits_with, no_ids)
                by = _logsumexp_pool(logits_wo,  yes_ids)
                bn = _logsumexp_pool(logits_wo,  no_ids)
                m_with = (ay - an)
                m_wo   = (by - bn)
                delta  = (m_with - m_wo)

                tau_margin = getattr(logits_processor_list[0], "tau_margin", None) if logits_processor_list else None
                penalty_value = getattr(logits_processor_list[0], "penalty_value", None) if logits_processor_list else None
                trigger = (delta < tau_margin) if (tau_margin is not None) else torch.zeros_like(delta, dtype=torch.bool)

                ay_np = ay.detach().cpu().numpy()
                an_np = an.detach().cpu().numpy()
                by_np = by.detach().cpu().numpy()
                bn_np = bn.detach().cpu().numpy()
                m_with_np = m_with.detach().cpu().numpy()
                m_wo_np = m_wo.detach().cpu().numpy()
                delta_np = delta.detach().cpu().numpy()
                trigger_np = trigger.detach().cpu().numpy()

                # Pre/post pooled logits and shifts (core TAD mechanism)
                L_yes_pre_np  = L_yes_pre.detach().cpu().numpy()
                L_no_pre_np   = L_no_pre.detach().cpu().numpy()
                L_yes_post_np = L_yes_post.detach().cpu().numpy()
                L_no_post_np  = L_no_post.detach().cpu().numpy()
                s_no_pre_np   = s_no_pre.detach().cpu().numpy()
                s_no_post_np  = s_no_post.detach().cpu().numpy()
                delta_L_yes_np = (L_yes_post - L_yes_pre).detach().cpu().numpy()
                delta_s_np     = (s_no_post - s_no_pre).detach().cpu().numpy()
                p_no_pre_np    = p_no_pre
                p_no_post_np   = p_no_post
            else:
                ay_np = an_np = by_np = bn_np = m_with_np = m_wo_np = delta_np = trigger_np = None
                tau_margin = None
                penalty_value = None
        
        # Generate 
        with torch.no_grad():  # Ensure we're not tracking gradients for inference
            generate_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                length_penalty=1.0,
                early_stopping=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                logits_processor=logits_processor_list,
                temperature=None,
                top_p=None,
                top_k=None
            )
            generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

        # Decode responses 
        responses = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Process results
        # for qa_item, response in zip(valid_batch, responses):
        #     result_item = {
        #         "yes_no": extract_yes_no(response), 
        #         "label": qa_item["text"].lower(), 
        #         "response": response  # Keep the full response for analysis   
        #     }
        #     batch_results.append(result_item)

        # for ROC
        for i, (qa_item, response) in enumerate(zip(valid_batch, responses)):
            result_item = {
                "yes_no": extract_yes_no(response),
                "label": qa_item["text"].lower(),
                "response": response,

                # ROC：P(no)
                "p_no": float(p_no[i]),
                "p_no_pre": float(p_no_pre_np[i]) if p_no_pre_np is not None else None,
                "p_no_post": float(p_no_post_np[i]) if p_no_post_np is not None else None,
                "L_yes_pre": float(L_yes_pre_np[i]) if L_yes_pre_np is not None else None,
                "L_no_pre": float(L_no_pre_np[i]) if L_no_pre_np is not None else None,
                "L_yes_post": float(L_yes_post_np[i]) if L_yes_post_np is not None else None,
                "L_no_post": float(L_no_post_np[i]) if L_no_post_np is not None else None,
                "s_no_pre": float(s_no_pre_np[i]) if s_no_pre_np is not None else None,
                "s_no_post": float(s_no_post_np[i]) if s_no_post_np is not None else None,
                "delta_L_yes": float(delta_L_yes_np[i]) if delta_L_yes_np is not None else None,
                "delta_s": float(delta_s_np[i]) if delta_s_np is not None else None,
                # Evidence terms (first-step)
                "ay": float(ay_np[i]) if ay_np is not None else None,
                "an": float(an_np[i]) if an_np is not None else None,
                "by": float(by_np[i]) if by_np is not None else None,
                "bn": float(bn_np[i]) if bn_np is not None else None,
                "m_with": float(m_with_np[i]) if m_with_np is not None else None,
                "m_wo": float(m_wo_np[i]) if m_wo_np is not None else None,
                "delta": float(delta_np[i]) if delta_np is not None else None,
                "trigger": bool(trigger_np[i]) if trigger_np is not None else None,
                "tau_margin": float(tau_margin) if tau_margin is not None else None,
                "penalty_value": float(penalty_value) if penalty_value is not None else None,
            }
            batch_results.append(result_item)
        
        # Clean up GPU memory
        del inputs
        if use_logits_processor:
            del inputs_without_audio
            del input_embeddings
            del logits_processor
        del generate_ids
        torch.cuda.empty_cache()
        
        return batch_results
        
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return []

def main():
    print(f"Starting evaluation sequentially.")
    print(f"Processing file: {filename}")
    
    # Load test data 
    with open(filename, 'r') as f:
        qa_pairs = json.load(f)

    model, processor = load_model_and_processor(model_flag)

    if dataset_name == "audiocaps_hallucination":
        all_results = []
        
        # Create progress bar for overall processing
        total_items = len(qa_pairs) 
        pbar = tqdm(total=total_items, desc="Processing QA pairs")
        processed_count = 0
        
        # Process batches sequentially
        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:min(i + batch_size, len(qa_pairs))]
            
            # Convert to conversation format  
            conversations = [
                [{
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": qa_item["path"]},
                        {"type": "text", "text": prefix_prompt + qa_item["Q"]}
                    ]
                }] for qa_item in batch
            ]
            
            # Prepare text inputs
            texts = [
                processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                for conv in conversations
            ]
            
            # Load audio files sequentially
            sampling_rate = processor.feature_extractor.sampling_rate 
            audios = [] 
            zero_audios = [] 
            valid_batch = [] 
            valid_texts = [] 
            
            for idx, qa_item in enumerate(batch):
                result = load_audio(qa_item, sampling_rate) # {"audio", "zero_audio", "qa_item", "sucess"}
                if result["success"]:
                    audios.append(result["audio"]) 
                    valid_batch.append(result["qa_item"]) 
                    valid_texts.append(texts[idx]) 
                    if use_logits_processor:
                        zero_audios.append(np.zeros_like(result["audio"])) 
            
            if not audios:
                continue
            
            # Process this batch
            batch_data = (valid_batch, audios, valid_texts, zero_audios)
            results = process_batch(batch_data, model, processor)
            all_results.extend(results)
            processed_count += len(results)
            pbar.update(len(results))
            
            # Clean up memory after each batch
            gc.collect()
            torch.cuda.empty_cache()
    
        pbar.close()

    elif dataset_name == "clotho_aqa":
        # Load the dataset 
        # {'clotho_aqa_test_filtered': ['audio', 'question', 'answer'], 'clotho_aqa_val_filtered': ['audio', 'question', 'answer']}
        dataset = load_dataset("lmms-lab/ClothoAQA") 
        test_ds = dataset["clotho_aqa_test_filtered"]
        val_ds = dataset["clotho_aqa_val_filtered"]
        full_ds = concatenate_datasets([test_ds, val_ds])
        # full_ds = full_ds.select(range(30))  # select 30 samples for test

        sampling_rate = processor.feature_extractor.sampling_rate 
        full_ds = full_ds.cast_column("audio", Audio(sampling_rate=sampling_rate)) 
    
        all_results = []
        print("Running batched inference on Clotho AQA test filtered")

        # Create progress bar for overall processing
        total_items = len(full_ds) 
        pbar = tqdm(total=total_items, desc="Processing QA pairs")
        processed_count = 0
        
        for start in tqdm(range(0, len(full_ds), batch_size)):
            end = min(start + batch_size, len(full_ds))
            # dict: {"audio": [...], "question": [...], "answer": [...]}
            batch = full_ds[start:end]
    
            audios_loaded = [a["array"] for a in batch["audio"]]  # list of np.array
            questions = batch["question"]                  # list of str
            answers = [a.lower() for a in batch["answer"]] # list of "yes"/"no"
    
            # chat template
            conversations = [
                [{
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": "placeholder"},
                        {"type": "text", "text": prefix_prompt + q},
                    ],
                }]
                for q in questions
            ]
    
            texts = [
                processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False,)
                for conv in conversations
            ]
            
            # Load audio files sequentially
            audios = [] 
            zero_audios = [] 
            valid_batch = [] 
            valid_texts = [] 
            
            for idx in range(len(batch["audio"])):
                audios.append(audios_loaded[idx])
                valid_batch.append({"text": answers[idx]})
                valid_texts.append(texts[idx])
                if use_logits_processor:
                    zero_audios.append(np.zeros_like(audios[idx])) 

            # Process this batch
            batch_data = (valid_batch, audios, valid_texts, zero_audios)
            results = process_batch(batch_data, model, processor)
            all_results.extend(results)
            processed_count += len(results)
            pbar.update(len(results))
            
            # Clean up memory after each batch
            gc.collect()
            torch.cuda.empty_cache()
    
        pbar.close()    

    else:
        raise ValueError("Invalid dataset name.")
    
    print("\nCalculating metrics...")
    # Calculate and display metrics
    acc, precision, recall, f1, yes_ratio, not_answer_ratio = discriminative_metric(all_results)

    # Date and time
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = filename.split("/")[-1].split(".")[0]
    plot_confusion_matrix(all_results, fig_name=f"{model_flag}_{use_logits_processor}_{dataset_name}_{alpha}_{fname}_{date_time}")
    
    # Create benchmark results dictionary
    benchmark_results = {
        "metrics": {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "yes_ratio": yes_ratio
        }
    }
    
    # Save detailed results with benchmark metrics
    final_results = {
        "benchmark_metrics": benchmark_results,
        "data_set": filename,
        "use_cad": use_logits_processor,
        "cad_alpha": alpha,
        "prefix_prompt": prefix_prompt,        
        "processed_items": processed_count,
        "total_items": total_items
    }
    
    # Save all results
    os.makedirs("./results", exist_ok=True)
    result_filename = f"./results/evaluation_results_{date_time}.json"
    with open(result_filename, 'w') as f:
        json.dump(final_results, f, indent=2)

    # ===== Save ROC arrays =====
    y_true = np.array([1 if r["label"] == "no" else 0 for r in all_results], dtype=np.int64)
    y_score = np.array([r["p_no"] for r in all_results], dtype=np.float32)

    os.makedirs("./roc_data", exist_ok=True)
    roc_filename = f"./roc_data/roc_{model_flag}_tad{int(use_logits_processor)}_{dataset_name}_alpha{alpha}_{date_time}.npz"

    np.savez(
        roc_filename,
        y_true=y_true,
        y_score=y_score,
        model_flag=model_flag,
        use_logits_processor=use_logits_processor,
        dataset_name=dataset_name,
        alpha=alpha,
        filename=filename,
        prefix_prompt=prefix_prompt,
    )

    print(f"ROC arrays saved to: {roc_filename}")
    # ===========================

    # ===== Save Evidences =====
    os.makedirs("./evidence_data", exist_ok=True)
    evidence_filename = f"./evidence_data/evidence_{model_flag}_tad{int(use_logits_processor)}_{dataset_name}_alpha{alpha}_{date_time}.npz"

    import numpy as _np
    def _arr(key):
        return _np.array([r.get(key, _np.nan) if r.get(key, None) is not None else _np.nan for r in all_results], dtype=_np.float64)

    ay = _arr("ay"); an = _arr("an"); by = _arr("by"); bn = _arr("bn")
    m_with = _arr("m_with"); m_wo = _arr("m_wo"); delta = _arr("delta")
    p_no_all = _np.array([r.get("p_no", _np.nan) for r in all_results], dtype=_np.float64)
    p_no_pre_all = _arr("p_no_pre"); p_no_post_all = _arr("p_no_post")
    L_yes_pre_all = _arr("L_yes_pre"); L_no_pre_all = _arr("L_no_pre")
    L_yes_post_all = _arr("L_yes_post"); L_no_post_all = _arr("L_no_post")
    s_no_pre_all = _arr("s_no_pre"); s_no_post_all = _arr("s_no_post")
    delta_L_yes_all = _arr("delta_L_yes"); delta_s_all = _arr("delta_s")
    trigger = _np.array([int(bool(r.get("trigger", False))) if r.get("trigger", None) is not None else -1 for r in all_results], dtype=_np.int64)
    label = _np.array([str(r.get("label", "")) for r in all_results])
    yes_no = _np.array([str(r.get("yes_no", "")) for r in all_results])

    _np.savez(
        evidence_filename,
        ay=ay, an=an, by=by, bn=bn,
        m_with=m_with, m_wo=m_wo, delta=delta,
        p_no=p_no_all,
        p_no_pre=p_no_pre_all,
        p_no_post=p_no_post_all,
        L_yes_pre=L_yes_pre_all,
        L_no_pre=L_no_pre_all,
        L_yes_post=L_yes_post_all,
        L_no_post=L_no_post_all,
        s_no_pre=s_no_pre_all,
        s_no_post=s_no_post_all,
        delta_L_yes=delta_L_yes_all,
        delta_s=delta_s_all,
        trigger=trigger,
        label=label,
        yes_no=yes_no,
        y_true=y_true,
        y_score=y_score,
        model_flag=model_flag,
        alpha=alpha,
        dataset_name=dataset_name,
        date_time=date_time,
    )
    print(f"Evidence arrays saved to: {evidence_filename}")
    # ===========================
    
    print(f"\nProcessed {processed_count} items out of {total_items} total items")
    print(f"Results saved to {result_filename}")

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    import argparse
    parser = argparse.ArgumentParser(description='Model configuration parameters')
    parser.add_argument('--model_flag', type=int, default=1, help='Model flag')
    parser.add_argument('--use_logits_processor', type=str2bool, default=True, help='Whether to use logits processor')
    parser.add_argument('--dataset_name', type=str, default="audiocaps_hallucination", help='audiocaps_hallucination or clotho_aqa')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter value')
    parser.add_argument('--filename', type=str, default="./data/sample-random.json", help='Data file path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Max new tokens generated')
    args = parser.parse_args()

    print("parameter settings:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    model_flag = args.model_flag
    use_logits_processor = args.use_logits_processor
    dataset_name = args.dataset_name
    alpha = args.alpha
    filename = args.filename
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens

    # for default and AAD
    # prefix_prompt = "Focus on the given audio and answer the following question. "
    # for TAD
    prefix_prompt = "Focus on the given audio and answer the following question. Answer in the format: 'Yes, ...' or 'No, ...', and always start with 'Yes' or 'No'. "
    torch.manual_seed(42)
    
    main()
