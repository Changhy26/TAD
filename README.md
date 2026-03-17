# TAD

Code for the paper: **TAD: Token-Adaptive Contrastive Decoding with Confidence-Guided Gating for Hallucination Mitigation in Large Audio-Language Models.**

## Overview

This repository provides the implementation of **TAD (Token-Adaptive Decoding)**, a training-free decoding method for mitigating hallucinations in large audio-language models (LALMs), especially in binary audio question answering tasks.

TAD is designed to reduce unsupported affirmative responses (e.g., answering "yes" when the queried sound is actually absent). The main idea is to contrast model logits under **real audio** and a **matched silent reference**, and then apply a **confidence-guided gating mechanism** at the first decoding step to suppress unreliable affirmative predictions when audio evidence is insufficient.

Our method is simple, plug-and-play, and does not require any parameter updates. It can be used as an inference-time decoding strategy for improving the reliability of audio-grounded decisions. The paper evaluates TAD on hallucination-related benchmarks including AudioCaps-Hallucination and Clotho-AQA}

## 🔔 Note

This paper is currently under review for **Interspeech 2026**.  
To comply with the double-blind review policy, some details may be anonymized or temporarily omitted. We will update this repository with full information after the review process is completed.

## Acknowledgement

This project is partially inspired by and builds upon the implementation of **Audio-Aware Decoding**. We sincerely thank the authors for releasing their code and resources, which provide valuable insights for our work.

- **Project Repository**: https://github.com/kuan2jiu99/audio-hallucination/tree/main/interspeech2024  
- **Paper**: https://arxiv.org/pdf/2506.07233  

The original work proposes an audio-aware logit processing method to mitigate hallucination issues in audio language models. Parts of our implementation are adapted and extended based on their open-source codebase.

## Dataset

We also make use of the dataset and audio resources provided by the original repository:

- **Dataset**: https://github.com/kuan2jiu99/audio-hallucination/tree/main/interspeech2024  
- **Audio Files**: https://drive.google.com/drive/folders/1S0IJxDTkP3Mi0Cj6CncOoYens2I--C1x?usp=sharing  

Please refer to the original repository for detailed descriptions and usage instructions of the data.

## Citation

If you find their work useful, please consider citing the original paper:

```bibtex
@article{audio_aware_decoding,
  title={Audio-Aware Decoding for Reducing Hallucinations in Audio Language Models},
  author={...},
  journal={arXiv preprint arXiv:2506.07233},
  year={2025}
}
