import argparse
SAMPLING_FREQ = 16000
from ctc_forced_aligner import ( # https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)
import json
import os
import polars as pl
from huggingface_hub import HfFileSystem
import torchaudio
import torch
from lcasr.utils.audio_tools import to_spectogram
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fs = HfFileSystem()
import resource
import time

def set_memory_limit(max_mb:int):
    """Sets a soft memory limit on the current process."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # Convert MB to Bytes
    resource.setrlimit(resource.RLIMIT_AS, (max_mb * 1024 * 1024, hard))

def load_audio(audio, dtype: torch.dtype, device: str):
    waveform, audio_sf = torchaudio.load(audio)  # waveform: channels X T
    if len(waveform.shape) == 2:
      waveform = torch.mean(waveform, dim=0)
    if audio_sf != SAMPLING_FREQ:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=audio_sf, new_freq=SAMPLING_FREQ
        )
    waveform = waveform.to(dtype).to(device)
    return waveform

def get_all_parquets(dataset_path = "datasets/espnet/floras/monolingual/train-*.parquet"):
  files = fs.glob(dataset_path)
  parquets = [el.split("/")[-1] for el in files]
  return parquets


def read_parquet(id, retries=10, delay_s=120):
    retry_count = 0
    while retry_count < retries:
      try:
        split = f'monolingual/{id}'
        df = pl.read_parquet('hf://datasets/espnet/floras/' + split)
        return df
      except Exception as e:
          logger.error(f"Error reading parquet {id}: {e}. Retrying in {delay_s} seconds...")
          time.sleep(delay_s)
          retry_count += 1
    raise Exception(f"Failed to read parquet {id} after {retries} retries.")

def align_sample(row, alignment_model, alignment_tokenizer, bsz=8, device='cuda'):
  text = row["text"][0]
  audio_bytes = row["audio"][0]["bytes"]
  audio = load_audio(audio_bytes, dtype=torch.float16 if device == "cuda" else torch.float32, device=device)
  tokens_starred, text_starred = preprocess_text(
      text,
      romanize=True,
      language="eng",
  )
  emissions, stride = generate_emissions(
      alignment_model, audio, batch_size=bsz
  )
  segments, scores, blank_token = get_alignments(
      emissions,
      tokens_starred,
      alignment_tokenizer,
  )
  spans = get_spans(tokens_starred, segments, blank_token)
  word_timestamps = postprocess_results(text_starred, spans, stride, scores)
  return word_timestamps, audio

def save_spec(spec, save_path, row_id):
   assert os.path.exists(os.path.dirname(save_path)), f"Directory {os.path.dirname(save_path)} does not exist."
   audio_root = os.path.join(save_path, "audio")
   assert os.path.exists(audio_root), f"Directory {audio_root} does not exist."
   audio_path = os.path.join(audio_root, f"{row_id}.pt")
   torch.save(spec, audio_path)

def save_text(data, save_path, row_id, duration):
    assert os.path.exists(os.path.dirname(save_path)), f"Directory {os.path.dirname(save_path)} does not exist."
    json_root = os.path.join(save_path, "text")
    assert os.path.exists(json_root), f"Directory {json_root} does not exist."
    data = {"word_timestamps": data, "duration": duration}
    json_path = os.path.join(json_root, f"{row_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def sign_completion(save_path, parquet_id):
    completion_file = os.path.join(save_path, f"signed_completions.txt")
    with open(completion_file, "a") as f:
        f.write(f"{parquet_id}\n")

def check_completion(save_path, parquet_id):
    completion_file = os.path.join(save_path, f"signed_completions.txt")
    if not os.path.exists(completion_file):
        return False
    with open(completion_file, "r") as f:
        completed = f.read().splitlines()
    return str(parquet_id) in completed

def get_duration_from_waveform(waveform, sampling_rate=16000):
    num_samples = waveform.shape[-1]
    duration_seconds = num_samples / sampling_rate
    return duration_seconds

def main(device, start, end, save_path):
    logger.info(f"Loading alignment model on device {device}...")
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    logger.info("Alignment model loaded.")
    logger.info(f"fetching full list of parquets...")
    parquets = get_all_parquets()
    parquets = parquets[start:end]
    logger.info(f"Processing parquets from index {start} to {end}...")

    for i, parquet in enumerate(parquets):
        parquet_id = i + start
        if check_completion(save_path, parquet_id):
            logger.info(f"Skipping already completed parquet {parquet} ({parquet_id})...")
            continue
        logger.info(f"Reading parquet {parquet} ({parquet_id})...")
        df = read_parquet(parquet)
        errors = 0
        for row in range(len(df)):
            try:
                logger.info(f"Processing row {row} out of {len(df)} in parquet {parquet} ({parquet_id})...")
                timestamps, audio = align_sample(df[row], alignment_model, alignment_tokenizer, device=device)
                duration = get_duration_from_waveform(audio, sampling_rate=SAMPLING_FREQ)
                spectrogram = to_spectogram(audio.cpu().float())
                row_id = f'{parquet_id}_{row}'
                save_spec(spectrogram.half(), save_path, row_id)
                save_text(timestamps, save_path, row_id, duration)
                logger.info(f"Completed: row {row} out of {len(df)} in parquet {parquet} ({parquet_id})")
            except MemoryError:
                logger.error(f"MemoryError encountered while processing row {row} in parquet {parquet} ({parquet_id}). Skipping this row.")
                errors += 1
            except Exception as e:
                logger.error(f"Error processing row {row} in parquet {parquet} ({parquet_id}): {e}. Skipping this row.")
                errors += 1
            finally:
                if errors >= 10:
                    logger.error(f"Encountered {errors} errors in parquet {parquet} ({parquet_id}). Killing the process, please debug.")
                    raise Exception(f"Too many errors (10) in parquet {parquet} ({parquet_id}).")  
        logger.info(f"Completed: parquet {parquet} ({parquet_id}) out of {len(parquets)} - encountered {errors} errors.")
        if errors == 0:
            sign_completion(save_path, parquet_id)
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
         "--device",
         type=str,
         default="cuda" if torch.cuda.is_available() else "cpu",
         help="Device to run the alignment model on.",
    )
    parser.add_argument(
         "--start",
         type=int,
         default=0,
         help="Start index of the parquet files to process.",
    )
    parser.add_argument(
         "--end",
         type=int,
         default=1,
         help="End index of the parquet files to process.",
    )
    parser.add_argument(
         "--save_path",
         type=str,
         default="/mnt/parscratch/users/acp21rjf/floras50aligned/", 
         help="Path to save the aligned results.",
    )
    parser.add_argument(
        "--max_cpu_mem",
        type=int,
        default=100,
        help="Maximum CPU memory (in GB) to use during processing. Used to catch/avoid kills due to OOM.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "audio"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "text"), exist_ok=True)

    set_memory_limit(args.max_cpu_mem * 1024) 

    main(args.device, args.start, args.end, args.save_path)

