import torch, torchaudio
import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from lib import get_all_parquets, read_parquet, load_audio
import os

def save_raw_text(text, save_path, split, parquet_id, row_id):
    text_path = os.path.join(save_path, split, "text", f"{parquet_id}_{row_id}.txt")
    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

def save_waveform(audio, save_path, split, parquet_id, row_id):
    audio_path = os.path.join(save_path, split, "audio", f"{parquet_id}_{row_id}.wav")
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    torchaudio.save(audio_path, audio.reshape(1, -1), 16000)

def main(split, save_path):
    logger.info(f"fetching full list of parquets...")
    dataset_path = f"datasets/espnet/floras/monolingual/{split}-*.parquet"
    parquets = get_all_parquets(dataset_path=dataset_path)
    
    logger.info(f"Processing parquets {len(parquets)} from split {split}...")

    for i, parquet in enumerate(parquets):
        parquet_id = i 
        logger.info(f"Reading parquet {parquet} ({parquet_id})...")
        df = read_parquet(parquet)
    
        for row in range(len(df)):
       
            logger.info(f"Processing row {row} out of {len(df)} in parquet {parquet} ({parquet_id})...")
            current_row = df[row]
            text = current_row["text"][0]
            audio_bytes = current_row["audio"][0]["bytes"]
            audio = load_audio(audio_bytes, dtype=torch.float32, device="cpu")

            save_raw_text(text, save_path, split, parquet_id, row)
            save_waveform(audio, save_path, split, parquet_id, row)

        logger.info(f"Completed: parquet {parquet} ({parquet_id}) out of {len(parquets)}.")
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
         "--save_path",
         type=str,
         default="/mnt/parscratch/users/acp21rjf/floras50_eval/", 
         help="Path to save the aligned results.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "dev", "train"],
    )
    args = parser.parse_args()

    main(split=args.split, save_path=args.save_path)

