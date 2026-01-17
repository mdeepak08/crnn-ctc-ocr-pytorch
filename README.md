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

## Current architecture explained with example
Think of one example cropped word image: **"cafe"**.

### What are B, C, H, W, F, T?
- **B**: batch size (number of images processed together). Example: batch_size=64 → **B=64**. For a single image → **B=1**.
- **C**: channels.
  - Input is grayscale → **C=1**
  - After CNN, channels become learned feature channels (e.g., **64/128/256**).
- **H, W**: height/width in pixels of the preprocessed image (commonly **H=32**, **W=128** due to resize + padding).
- **T**: number of time steps after CNN (roughly the **CNN output width** \(W'\)). It’s like “how many vertical slices the model sees” left→right.
- **F**: feature size per time step (how many numbers describe one slice). Often **F ≈ cnn_out_channels** (e.g., **256**).

### 1) CNN: image → feature map
Input (one image):
- Shape: **[B, 1, 32, 128]** → **[1, 1, 32, 128]**

After CNN (typical example):
- Shape: **[B, C_feat, H', W']** → **[1, 256, 1, 32]**
  - Width **128** gets downsampled by ~4 → **32** columns (time steps).

**What are those 256 numbers?**  
At each column position along the word, the CNN outputs a **256-dim vector** describing visual patterns it learned (strokes/curves/background boundaries, etc.). So the word becomes a **strip of feature vectors across width**.

### 2) Make it a sequence: feature map → [T, B, F]
We treat the CNN output width \(W'\) as time:
- From **[1, 256, 1, 32]** → squeeze \(H'\) and rearrange → **[T, B, F] = [32, 1, 256]**

So for "cafe", the model now has **32 feature vectors**, each representing a left→right “slice” of the word.

### 3) BiLSTM: add context across the slices
Each time step has a feature vector \(x_t \in \mathbb{R}^{256}\).

A **BiLSTM** reads this sequence in both directions:
- **Forward LSTM** (left → right): builds a hidden state \(h^{→}_t\) summarizing what came before.
- **Backward LSTM** (right → left): builds \(h^{←}_t\) summarizing what comes after.
- They’re concatenated: \(h_t = [h^{→}_t;\,h^{←}_t]\) so the output size becomes **2H** (example: hidden=256 → output features per step = 512).

**What “context” does it learn?**  
It helps resolve ambiguous local shapes by looking at neighbors:
- The middle of **"m"** can resemble **"rn"** in blurry text; neighbors help disambiguate.
- **"i" vs "l" vs "1"** depends on surrounding strokes and spacing.
- Slices between characters are mostly background; context helps treat them as “blank-like”.

### 4) Linear + log-softmax: per-time-step character probabilities
For each time step \(t\), the model outputs a distribution over:
- vocabulary characters (a–z, 0–9, etc.)
- plus the **CTC blank** (index 0)

So `log_probs` has shape:
- **[T, B, num_classes]**
- Example: **[32, 1, 37]** (36 chars + blank)

### CTC explained again (with greedy + beam)
**CTC = Connectionist Temporal Classification**.

CTC is the trick that lets the model output **T predictions** (like 32) even though the true word has fewer characters (like 4), **without** telling it exactly which time step aligns to which character.

CTC introduces a special **blank** token `_` and defines decoding as:
1) **collapse repeats** (only consecutive repeats)
2) **remove blanks**

#### Greedy decoding example
Pick the argmax class at each time step, then apply the CTC rules.

Raw per-step argmax (shortened):
`_ _ c c _ a a _ f _ e e _ _`

1) Collapse repeats:
`_ c _ a _ f _ e _`

2) Remove blanks:
`cafe`

Greedy can fail if the model makes one locally-wrong argmax that ruins the whole word.

#### Beam decoding example (why it can help)
Beam search keeps the top **K** most probable partial strings as it scans time steps.

If the model is uncertain at one step (e.g., between `e` and `c`):
- Greedy commits immediately (might pick the wrong one).
- Beam keeps multiple hypotheses alive (example: `"caf"` and `"cac"`) and later evidence can make `"cafe"` win.

Note: A proper CTC beam search tracks blank/non-blank endings because repeats behave differently; our code uses a prefix beam search.

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

### After fine-tuning on IIIT-5K + train-time augmentations
Greedy decoding only (augmentation was enabled during training; evaluation is clean/no-aug):

| Dataset | Split | Decoder | CER ↓ | Word Acc ↑ |
|---|---|---|---:|---:|
| IIIT-5K | test | greedy | 0.0994 | 0.7577 |
| SVT | test | greedy | 0.1396 | 0.6615 |

### After training on mixed MJSynth + IIIT-5K (with augmentations)
Greedy decoding only:

| Dataset | Split | Decoder | CER ↓ | Word Acc ↑ |
|---|---|---|---:|---:|
| IIIT-5K | test | greedy | 0.0976 | 0.7447 |
| SVT | test | greedy | 0.1225 | 0.7002 |

### After mixed training + short IIIT-only fine-tune (2 epochs)
Greedy decoding only:

| Dataset | Split | Decoder | CER ↓ | Word Acc ↑ |
|---|---|---|---:|---:|
| IIIT-5K | test | greedy | 0.0921 | 0.7680 |
| SVT | test | greedy | 0.1307 | 0.6878 |

## Notes / next upgrades
- To enable train-time augmentations, set `augment.enabled: true` in your config (validation/test are always run without augmentation).
- Add augmentation (blur, perspective, noise) and stronger CNN backbones
- Add an LM-aware decoding (beam + char/word LM)
- Replace BiLSTM with a Transformer encoder

