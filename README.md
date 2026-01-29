uses MMS CTC alignment (https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) to preprocess floras-50 data (https://huggingface.co/datasets/espnet/floras) for ASR training 

requires install of https://github.com/robflynnyh/long-context-asr (just for spectrogram computation) to run
and also install https://github.com/MahmoudAshraf97/ctc-forced-aligner.git (for force alignment)

file explanation:

align_and_save.py
- Where alignment is performed
- processes a portion of the total parquets (based on args)
- saves dictionary containing word-level alignments of text, and forms spectrogram of audio and saves as .pt file

create_launch_scripts.py
- For parrellizing align_and_save.py on sheffield HPC, we input total GPUs to use and it creates several slurm launch files where when launched will split the parquets over these jobs
- run with -l or --launch to automatically launch the slurm jobs on creation

create_mapping.py
- to be ran after alignment is complete
- creates a dictionary containing sample id and the corresponding spec.pt (audio) and alignment (text) paths
- Also stores the duration (to help efficient batch creation during training)
- the completed file can then be used by a dataloader for ASR training

print_unique_completions.py
- looks in target directory and counts number of parquets that have been logged as complete out of total (900)

all files use default path that are unique to my dir on sheffield HPC, but can be configured otherwise via args
