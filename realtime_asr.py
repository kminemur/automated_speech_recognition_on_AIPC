import argparse
import queue
import sys
import time

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
        default=1.0,
        help="Chunk duration in seconds (default: 1.0).",
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
        default="<|en|>",
        help="Language token (default: <|en|>).",
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
                    print(f"[{elapsed:.2f}s] {text}")
    except KeyboardInterrupt:
        print("Stopping...")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
