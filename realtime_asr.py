import argparse
import queue
import sys
import time
import re
import collections
import datetime

import numpy as np
import sounddevice as sd
import openvino_genai as ov_genai
import webrtcvad


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
        help="Max segment duration in seconds (default: 2.5).",
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
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="VAD aggressiveness (0-3, higher is stricter).",
    )
    parser.add_argument(
        "--vad-frame-ms",
        type=int,
        default=20,
        choices=[10, 20, 30],
        help="VAD frame size in ms (10/20/30).",
    )
    parser.add_argument(
        "--vad-silence-ms",
        type=int,
        default=600,
        help="Silence duration to end segment (ms).",
    )
    parser.add_argument(
        "--vad-min-speech-ms",
        type=int,
        default=200,
        help="Minimum speech duration to start segment (ms).",
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

    max_segment_samples = int(args.seconds * args.samplerate)
    frame_samples = int(args.samplerate * args.vad_frame_ms / 1000)
    if max_segment_samples <= 0:
        print("Max segment duration must be > 0.", file=sys.stderr)
        return 2
    if frame_samples <= 0:
        print("VAD frame size must be > 0.", file=sys.stderr)
        return 2

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("Starting realtime ASR. Press Ctrl+C to stop.")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Max segment seconds: {args.seconds}")
    print(f"VAD aggressiveness: {args.vad_aggressiveness}")
    print(f"VAD frame ms: {args.vad_frame_ms}")
    print(f"VAD silence ms: {args.vad_silence_ms}")
    print(f"VAD min speech ms: {args.vad_min_speech_ms}")
    print(f"Sample rate: {args.samplerate}")

    buffer = np.empty((0,), dtype=np.float32)
    last_text = ""
    last_norm = ""
    in_speech = False
    silent_frames = 0
    speech_frames = 0
    segment_frames: list[np.ndarray] = []
    pre_frames = max(1, int(300 / args.vad_frame_ms))
    pre_buffer = collections.deque(maxlen=pre_frames)
    min_speech_frames = max(1, int(args.vad_min_speech_ms / args.vad_frame_ms))
    silence_end_frames = max(1, int(args.vad_silence_ms / args.vad_frame_ms))
    vad = webrtcvad.Vad(args.vad_aggressiveness)

    def emit_segment(segment: np.ndarray) -> None:
        nonlocal last_emit, last_text, last_norm
        if segment.size == 0:
            return
        result = pipe.generate(segment, generation_config=gen_config)
        now_ts = datetime.datetime.now().strftime("%H:%M:%S")
        last_emit = time.time()
        text = result.get("text", str(result)) if isinstance(result, dict) else str(result)
        text = postprocess_text(text)
        norm = normalize_sentence(text)
        if text and norm != last_norm:
            print(f"[{now_ts}] {text}")
            last_text = text
            last_norm = norm

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

                while buffer.shape[0] >= frame_samples:
                    frame = buffer[:frame_samples]
                    buffer = buffer[frame_samples:]

                    pre_buffer.append(frame)
                    pcm16 = (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16)
                    is_speech = vad.is_speech(pcm16.tobytes(), args.samplerate)

                    if is_speech:
                        speech_frames += 1
                        silent_frames = 0
                        if not in_speech and speech_frames >= min_speech_frames:
                            in_speech = True
                            segment_frames = list(pre_buffer)
                        if in_speech:
                            segment_frames.append(frame)
                    else:
                        if in_speech:
                            segment_frames.append(frame)
                        silent_frames += 1
                        if not in_speech:
                            speech_frames = 0

                    if in_speech:
                        segment_samples = sum(f.shape[0] for f in segment_frames)
                        if segment_samples >= max_segment_samples:
                            emit_segment(np.concatenate(segment_frames))
                            segment_frames = []
                            silent_frames = 0

                        if silent_frames >= silence_end_frames and segment_frames:
                            emit_segment(np.concatenate(segment_frames))
                            segment_frames = []
                            pre_buffer.clear()
                            in_speech = False
                            silent_frames = 0
                            speech_frames = 0
    except KeyboardInterrupt:
        print("Stopping...")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
