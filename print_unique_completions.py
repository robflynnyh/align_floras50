import argparse, os



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
         "--save_path",
         type=str,
         default="/mnt/parscratch/users/acp21rjf/floras50aligned/", 
         help="Path to save the aligned results.",
    )
    parser.add_argument(
         "--max_completions",
            type=int,
            default=900
    )
    args = parser.parse_args()

    path = os.path.join(args.save_path, "signed_completions.txt")
    with open(path, "r") as f:
        completions = f.readlines()
    unique_completions = set(completions)
    print(f"Unique completions: {len(unique_completions)}/{args.max_completions}")