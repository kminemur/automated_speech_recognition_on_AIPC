import argparse
import queue
import sys
import time
import re

import numpy as np
import sounddevice as sd
import openvino_genai as ov_genai


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime Whisper ASR with 1s audio chunks using OpenVINO."
    )
    parser.add_argument(
        "--model",
        default="whisper-tiny-ov",
        help="Model directory (e.g., whisper-tiny-ov or whisper-kotoba-ov).",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        choices=["CPU", "GPU", "NPU", "AUTO"],
        help="OpenVINO device to use (default: NPU).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=2.5,
        help="Chunk duration in seconds (default: 2.5).",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Sampling rate in Hz (default: 16000).",
    )
    parser.add_argument(
        "--mic",
        type=int,
        default=None,
        help="Input device index. Leave empty for default device.",
    )
    parser.add_argument(
        "--language",
        default="<|ja|>",
        help="Language token (default: <|ja|>).",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task mode (default: transcribe).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    cache_config = {}
    if args.device in ["GPU", "NPU"]:
        cache_config["CACHE_DIR"] = "whisper_cache"

    pipe = ov_genai.WhisperPipeline(args.model, args.device, **cache_config)

    gen_config = pipe.get_generation_config()
    gen_config.language = args.language
    gen_config.task = args.task

    q: queue.Queue[np.ndarray] = queue.Queue()

    chunk_samples = int(args.seconds * args.samplerate)
    if chunk_samples <= 0:
        print("Chunk duration must be > 0.", file=sys.stderr)
        return 2

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("Starting realtime ASR. Press Ctrl+C to stop.")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Chunk seconds: {args.seconds}")
    print(f"Sample rate: {args.samplerate}")

    buffer = np.empty((0,), dtype=np.float32)
    last_text = ""
    last_norm = ""

    filler_words = [
        "えー", "ええと", "えっと", "あの", "あのー", "その", "そのー",
        "まー", "まぁ", "うーん", "ええ", "うん", "えっとー",
    ]
    banned_phrases = [
        "ご視聴ありがとうございました。",
        "ご視聴ありがとうございました",
    ]
    filler_pattern = re.compile(
        r"(?:^|\s)(?:" + "|".join(map(re.escape, filler_words)) + r")(?:$|\s)"
    )
    banned_pattern = re.compile("|".join(map(re.escape, banned_phrases)))

    def compress_words(text: str) -> str:
        parts = text.split()
        if not parts:
            return ""
        out = [parts[0]]
        for token in parts[1:]:
            if token != out[-1]:
                out.append(token)
        return " ".join(out)

    def postprocess_text(text: str) -> str:
        cleaned = banned_pattern.sub(" ", text)
        cleaned = filler_pattern.sub(" ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = compress_words(cleaned)
        return cleaned

    def normalize_sentence(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        normalized = re.sub(r"[。．\.！!？?]+$", "", normalized)
        return normalized

    try:
        with sd.InputStream(
            samplerate=args.samplerate,
            channels=1,
            dtype="float32",
            callback=audio_callback,
            device=args.mic,
            blocksize=0,
        ):
            last_emit = time.time()
            while True:
                data = q.get()
                if data.ndim > 1:
                    data = data[:, 0]
                buffer = np.concatenate((buffer, data))

                while buffer.shape[0] >= chunk_samples:
                    chunk = buffer[:chunk_samples]
                    buffer = buffer[chunk_samples:]

                    result = pipe.generate(chunk, generation_config=gen_config)
                    elapsed = time.time() - last_emit
                    last_emit = time.time()
                    text = result.get("text", str(result)) if isinstance(result, dict) else str(result)
                    text = postprocess_text(text)
                    norm = normalize_sentence(text)
                    if text and norm != last_norm:
                        print(f"[{elapsed:.2f}s] {text}")
                        last_text = text
                        last_norm = norm
    except KeyboardInterrupt:
        print("Stopping...")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
