# OCR Text Recognition with CRNN + CTC (PyTorch)

End-to-end **scene text word recognition**: given a cropped word image, predict the word using a **CRNN** (CNN → BiRNN → CTC loss).

## What you get
- **Reproducible** training/eval with YAML configs, deterministic seeds, checkpoints, and TensorBoard logs
- **Clean data pipeline** producing unified CSVs (`path,text`) + a saved `vocab.json`
- **Correct CRNN sequence modeling**: CNN feature map → time steps → BiLSTM/GRU → CTC
- **Evaluation**: CER + word accuracy, with **greedy** + **beam** CTC decoding
- **Device support**: Apple Silicon **MPS**, **CUDA**, or **CPU**
- **Demo**: Streamlit app (upload image → predicted text)

## Datasets
- **MJSynth / Synth90k** (train): `https://www.robots.ox.ac.uk/~vgg/data/text/`  
  (Optional HF mirror) `https://huggingface.co/datasets/priyank-m/MJSynth_text_recognition`
- **IIIT-5K** (eval): `https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset`
- **SVT** (eval): `https://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset`

## Repo layout
Key folders/files:
- `scripts/` for dataset preparation
- `data/raw/` and `data/processed/` are **git-ignored**
- `train.py`, `eval.py`, `infer.py`, `demo_app.py`

## Environment (Python 3.10+)
Create and activate a virtual environment:

```bash
cd /Users/deepakm/Documents/Deep_learning/RNN
# Recommended on macOS: use Python 3.11/3.12 (PyTorch wheels typically don’t support 3.13 yet).
# Example (Homebrew): brew install python@3.12
python3.12 -m venv .venv  # or: python3 -m venv .venv (if your python3 is 3.11/3.12)
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

PyTorch notes:
- On **macOS (Apple Silicon)**, `pip install torch torchvision` will install a build with **MPS** support.
- On **CUDA** machines, install the CUDA-enabled wheels per PyTorch’s official instructions.

## Quickstart on a tiny sanity subset
If you don’t have datasets downloaded yet, you can generate a tiny synthetic dataset first:

```bash
python scripts/make_toy_dataset.py --out_dir samples/toy --num_samples 200
python train.py --config configs/toy.yaml
python eval.py --config configs/toy.yaml --ckpt checkpoints/best_by_cer.pt --dataset_csv samples/toy/val.csv --decoder greedy
```

1) Prepare a small processed CSV + vocab (example uses MJSynth folder):

```bash
python scripts/prepare_mjsynth.py --raw_root data/raw/mjsynth --out_dir data/processed --max_len 25 --limit 2000
python scripts/make_vocab.py --csv_paths data/processed/mjsynth_train.csv data/processed/mjsynth_val.csv --out_path data/processed/vocab.json
```

2) Train:

```bash
python train.py --config configs/mjsynth_train.yaml
```

3) Evaluate (greedy / beam):

```bash
python eval.py --config configs/base.yaml --ckpt checkpoints/best_by_cer.pt --dataset mjsynth_val --decoder greedy
python eval.py --config configs/base.yaml --ckpt checkpoints/best_by_cer.pt --dataset mjsynth_val --decoder beam
```

4) Single-image inference:

```bash
python infer.py --ckpt checkpoints/best_by_cer.pt --vocab data/processed/vocab.json --image path/to/word.jpg --decoder greedy
```

## Streamlit demo

```bash
streamlit run demo_app.py
```

## Metrics
- **CER**: Levenshtein distance / max(1, len(gt))
- **Word accuracy**: exact-match ratio

## Results (current)
All results below are **unconstrained** (no lexicon/LM).

Important note on checkpoints:
- `checkpoints/best_by_cer.pt` can be **overwritten** between runs (e.g., pretrain vs fine-tune). If you want to keep both, copy/rename it after each run.

### Pretrain on MJSynth (subset)
- **MJSynth subset**: train=190,000 / val=10,000 (sampled from HF mirror)
- **best epoch (by MJSynth val CER)**: epoch 11 (val CER 0.0872, val word acc 0.6665)

| Dataset | Split | Decoder | CER ↓ | Word Acc ↑ |
|---|---|---|---:|---:|
| MJSynth (subset) | val | greedy | 0.0872 | 0.6665 |
| MJSynth (subset) | val | beam | 0.0981 | 0.6184 |
| IIIT-5K | test | greedy | 0.1971 | 0.4977 |
| IIIT-5K | test | beam | 0.2284 | 0.4223 |
| SVT | test | greedy | 0.1791 | 0.5100 |
| SVT | test | beam | 0.1963 | 0.4760 |

### After fine-tuning on IIIT-5K
Evaluated after fine-tuning on IIIT-5K (starting from the MJSynth-pretrained checkpoint):

| Dataset | Split | Decoder | CER ↓ | Word Acc ↑ |
|---|---|---|---:|---:|
| IIIT-5K | test | greedy | 0.1082 | 0.7360 |
| IIIT-5K | test | beam | 0.1095 | 0.7323 |
| SVT | test | greedy | 0.1607 | 0.6182 |
| SVT | test | beam | 0.1612 | 0.6167 |

### After IIIT-5K fine-tuning + augmentations
Same fine-tuning setup, but with train-time augmentations enabled (`augment.enabled: true`) and evaluated with greedy decoding:

| Dataset | Split | Decoder | CER ↓ | Word Acc ↑ |
|---|---|---|---:|---:|
| IIIT-5K | test | greedy | 0.0994 | 0.7577 |
| SVT | test | greedy | 0.1396 | 0.6615 |

## Notes / next upgrades
- To enable train-time augmentations, set `augment.enabled: true` in your config (validation/test are always run without augmentation).
- Add augmentation (blur, perspective, noise) and stronger CNN backbones
- Add an LM-aware decoding (beam + char/word LM)
- Replace BiLSTM with a Transformer encoder

