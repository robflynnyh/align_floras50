import polars as pl
import time
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
import logging
logger = logging.getLogger(__name__)
import torch
import torchaudio
SAMPLING_FREQ = 16000

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




