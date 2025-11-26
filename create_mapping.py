import argparse
import json
import sys
import logging
from pathlib import Path
import os
from tqdm import tqdm
from typing import Dict, Any

# 1. Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def open_json(text_file_json: Path) -> dict:
    with open(text_file_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data
    
def get_duration_from_text_file(text_file_json: Dict[str, Any]) -> float:
    return text_file_json["duration"]

def write_to_text_file(aggregated_text: str, output_path: Path) -> None:
    """ appends to an existing (or not yet created text file) """
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(aggregated_text + "\n")

def aggregate_text_from_json(text_file_json: Dict[str, Any]) -> None:
    all_text = " ".join([el["text"] for el in text_file_json["word_timestamps"]])
    save_path: Path = Path("./tmp/aggregated_text.txt").resolve()
    write_to_text_file(all_text, save_path)

def create_dataset_mapping(parent_dir_str: str, aggregate_text: bool = False) -> bool:
    """
    Scans the parent directory for audio/text pairs and writes a JSON mapping.
    Returns True if successful, False if directories are missing.
    """
    
    # Resolve absolute paths immediately
    parent_path = Path(parent_dir_str).resolve()
    audio_dir = parent_path / "audio"
    text_dir = parent_path / "text"

    # Validate Directories
    if not audio_dir.exists() or not text_dir.exists():
        logger.error(f"Could not find 'audio' or 'text' directories inside {parent_path}")
        return False

    mapping = {}
    logger.info(f"Scanning directory: {parent_path}")
    
    files_found = 0
    
    # Iterate through audio files
    for audio_file in tqdm(audio_dir.glob("*.pt"), total=len(list(audio_dir.glob("*.pt"))), desc="Processing audio files"): 
        file_id = audio_file.stem
        expected_text_file = text_dir / f"{file_id}.json"

        if expected_text_file.exists():
            # Store absolute paths as strings
            json_data = open_json(expected_text_file)

            if aggregate_text:
                aggregate_text_from_json(json_data)

            mapping[file_id] = {
                "audio": str(audio_file),
                "txt": str(expected_text_file),
                "duration": get_duration_from_text_file(json_data)
            }
            files_found += 1
        else:
            logger.warning(f"Match not found for ID '{file_id}' (missing text file)")

    # Write Output
            
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    output_path = Path("./tmp/mapping.json").resolve()
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=4)
        
        logger.info(f"Success! Mapped {files_found} pairs.")
        logger.info(f"Output saved to: {output_path}")
        return True

    except IOError as e:
        logger.error(f"Failed to write output file: {e}")
        return False

if __name__ == "__main__":
    # --- ARGUMENT PARSING HAPPENS HERE ---
    parser = argparse.ArgumentParser(
        description="Create a JSON mapping of audio (.pt) and text (.json) files."
    )
    parser.add_argument(        
        "--parent_dir", 
        type=str, 
        default="/mnt/parscratch/users/acp21rjf/floras50aligned/",
        help="The path to the parent directory containing 'audio' and 'text' subfolders."
    )
    parser.add_argument(
        "--aggregate_text",
        action="store_true",
        help="If set, aggregate all text into a single file, resulting file can be used for training tokenizers."
    )

    args = parser.parse_args()

    # --- PASS ARGS TO THE LOGIC FUNCTION ---
    success = create_dataset_mapping(args.parent_dir, args.aggregate_text)
    
    # Exit with status code 1 if the function failed (for CI/CD or shell scripts)
    if not success:
        sys.exit(1)