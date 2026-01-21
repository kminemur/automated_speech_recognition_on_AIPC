import io
import time
import argparse
import numpy as np
import soundfile as sf
import openvino_genai as ov_genai
from datasets import load_dataset

# Download models from Hugginface Hub
# whiper-tiny 37.8M
# kotoba-whisper 0.8B
# cmd: optimum-cli export openvino --trust-remote-code --model openai/whisper-tiny whisper-tiny-ov --disable-stateful
# cmd: optimum-cli export openvino --trust-remote-code --model kotoba-tech/kotoba-whisper-v2.2 whisper-kotoba-ov --disable-stateful
model_id = "whisper-tiny-ov"
#model_id = "whisper-kotoba-ov"

parser = argparse.ArgumentParser(description="Benchmark Whisper OpenVINO generation.")
parser.add_argument(
    "--device",
    default="NPU",
    choices=["CPU", "GPU", "NPU", "AUTO"],
    help="OpenVINO device to use (default: NPU)",
)
args = parser.parse_args()

# Set device to CPU, GPU or NPU
device = args.device

# cache config
cache_config = {}
if device in ["GPU", "NPU"]:
    cache_config["CACHE_DIR"] = "whisper_cache"
    
pipe = ov_genai.WhisperPipeline(model_id, device, **cache_config)

# Config
gen_config = pipe.get_generation_config()
gen_config.language = "<|en|>"  # set language to Japanese
gen_config.task = "transcribe"  # set task to translation

# load dataset
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.with_format("arrow")

# get audio date
sample_table = dataset[0:1]
audio_col = sample_table.column("audio")
audio_struct = audio_col[0].as_py()
audio_bytes = audio_struct["bytes"]

# decide audio data by soundfile
audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

# conver audio date to float32 numpy array
audio_array = audio_array.astype(np.float32)

# benchmark: 300 iterations
num_iterations = 300
print(f"Starting {num_iterations} iterations...")
print(f"Device: {device}")
start_time = time.time()

for i in range(num_iterations):
  result = pipe.generate(audio_array, generation_config=gen_config)
  if i % 50 == 0:
      print(f"Iteration {i}/{num_iterations} completed.")

end_time = time.time()
total_time = end_time - start_time
ave_time = total_time / num_iterations

print(f"Total time: {total_time} seconds")
print(f"Average time per iteration: {ave_time} seconds")
print(f"Result: {result}")
