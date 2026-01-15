## MJSynth / Synth90k download

- Hugging Face dataset card: `https://huggingface.co/datasets/priyank-m/MJSynth_text_recognition`

After download, set your folder like:
- `data/raw/mjsynth/` (contains the images + any annotation files like `annotation.txt`)

Then run:

```bash
python scripts/prepare_mjsynth.py --raw_root data/raw/mjsynth --out_dir data/processed --max_len 25
python scripts/make_vocab.py --csv_paths data/processed/mjsynth_train.csv data/processed/mjsynth_val.csv --out_path data/processed/vocab.json
```

