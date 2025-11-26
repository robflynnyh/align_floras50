from lcasr.utils.audio_tools import train_tokenizer
import os

if __name__ == "__main__":
    assert os.path.exists("./tmp/aggregated_text.txt"), "Please make sure the aggregated text file exists at ./tmp/aggregated_text.txt, create one using create_mapping.py with aggregate_text option." 

    train_tokenizer(
        raw_txt="./tmp/aggregated_text.txt",
        save_path="./tmp/"
    )