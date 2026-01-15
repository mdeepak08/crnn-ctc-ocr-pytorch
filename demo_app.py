from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.dataloaders.transforms import OCRPreprocess, OCRTransformConfig
from src.decoding.beam import BeamDecoderConfig, beam_decode
from src.decoding.greedy import GreedyDecoderConfig, greedy_decode
from src.models.crnn import CRNN, CRNNConfig
from src.utils.device import DeviceConfig, get_device


REPO_ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_model_and_vocab(ckpt_path: str, vocab_path: str, device_pref: str) -> tuple[CRNN, dict, torch.device]:
    device = get_device(DeviceConfig(prefer=device_pref))
    state = torch.load(ckpt_path, map_location="cpu")
    vocab = json.loads(Path(vocab_path).read_text())
    idx2char = list(vocab["idx2char"])
    num_classes = len(idx2char)

    cfg = state.get("config", {})
    model_cfg = cfg.get("model", {})
    mcfg = CRNNConfig(
        img_h=int(model_cfg.get("img_h", 32)),
        num_channels=int(model_cfg.get("num_channels", 1)),
        num_classes=num_classes,
        cnn_out_channels=int(model_cfg.get("cnn_out_channels", 256)),
        rnn_hidden=int(model_cfg.get("rnn_hidden", 256)),
        rnn_layers=int(model_cfg.get("rnn_layers", 2)),
        rnn_type=str(model_cfg.get("rnn_type", "lstm")),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    model = CRNN(mcfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, vocab, device


def main() -> None:
    st.set_page_config(page_title="CRNN OCR Demo", layout="centered")
    st.title("CRNN + CTC OCR (Word Recognition)")

    ckpt = st.text_input("Checkpoint path", value=str(REPO_ROOT / "checkpoints" / "best_by_cer.pt"))
    vocab = st.text_input("Vocab path", value=str(REPO_ROOT / "data" / "processed" / "vocab.json"))

    col1, col2 = st.columns(2)
    with col1:
        device_pref = st.selectbox("Device", options=["auto", "mps", "cuda", "cpu"], index=0)
    with col2:
        decoder = st.selectbox("Decoder", options=["greedy", "beam"], index=0)

    beam_width = st.slider("Beam width", min_value=2, max_value=20, value=5, step=1, disabled=(decoder != "beam"))

    img_h = st.number_input("Image height", min_value=16, max_value=64, value=32, step=1)
    img_w = st.number_input("Max image width", min_value=64, max_value=512, value=128, step=8)

    uploaded = st.file_uploader("Upload a cropped word image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("Upload a word image to run OCR.")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_container_width=True)

    try:
        model, vocab_obj, device = load_model_and_vocab(ckpt, vocab, device_pref)
        idx2char = list(vocab_obj["idx2char"])
        blank_idx = int(vocab_obj.get("blank_idx", 0))

        tfm = OCRPreprocess(OCRTransformConfig(img_h=int(img_h), img_w=int(img_w)))
        x, _ = tfm(img)
        x = x.unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            log_probs = model(x)
        if decoder == "beam":
            pred = beam_decode(
                log_probs, idx2char, BeamDecoderConfig(blank_idx=blank_idx, beam_width=int(beam_width))
            )[0]
        else:
            pred = greedy_decode(log_probs, idx2char, GreedyDecoderConfig(blank_idx=blank_idx))[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        st.subheader("Prediction")
        st.code(pred)
        st.caption(f"Device: {device} • Inference time: {dt_ms:.1f} ms")
    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()

