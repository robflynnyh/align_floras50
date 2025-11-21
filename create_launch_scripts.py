import argparse
import os
import shutil

run_strings = {
    'a100':f"""#!/bin/bash\n
#SBATCH --time=90:00:00
#SBATCH --mem=140GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main/

""",
    'h100':f"""#!/bin/bash\n
#SBATCH --time=96:00:00
#SBATCH --mem=110GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main/

"""
}

def get_script(start, end, gpu_type):
    script = f"""python align_and_save.py \\
    --device cuda \\
    --start {start} \\
    --end {end} \\
    --save_path /mnt/parscratch/users/acp21rjf/floras50aligned/
    """
    full_script = run_strings[gpu_type] + script
    return full_script


def main(args):
    num_gpus = args.num_gpus
    total_parquets = args.total_parquets
    gpu_type = args.gpu_type

    parquets_per_gpu = total_parquets // num_gpus
    remainder = total_parquets % num_gpus

    scripts = []
    start = 0
    for i in range(num_gpus):
        end = start + parquets_per_gpu + (0 if i != num_gpus - 1 else remainder)
        script = get_script(start, end, gpu_type)
        scripts.append((i, script))
        start = end

    paths = []
    for i, script in scripts:
        path = f"./tmp/launch_align_gpu_{i}.sh"
        with open(path, "w") as f:
            f.write(script)
        print(f"Created script: {path}")
        paths.append(path)

    if args.launch:
        for path in paths:
            os.system(f"sbatch {path}")
            print(f"Launched script: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create launch script for splitting alignment over multiple GPUs"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=11,
        help="Number of GPUs to split the alignment over.",
    )
    parser.add_argument(
        "--total_parquets",
        type=int,
        default=1500,
        help="Total number of parquet files to process.",
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="a100",
        choices=["a100", "h100"],
        help="Type of GPU to use for the job script.",
    )
    parser.add_argument(
        "-l", "--launch", action="store_true", help="Whether to launch the created scripts."
    )
    args = parser.parse_args()
    if os.path.exists('/tmp'):
        shutil.rmtree('./tmp', ignore_errors=True)
    os.makedirs("./tmp")
    main(args)

