import os

SEQS_DIR = "logs/3558f1342/render"
log_dir = "multi_process_img/logs/"


processed_seqs = set()

for log_file_name in os.listdir(log_dir):
    log_file_path = os.path.join(log_dir, log_file_name)
    seqs = set()
    with open(log_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            seq_name = line.split(".")[0]
            if seq_name.startswith("s"):
                seqs.add(seq_name)
    processed_seqs = processed_seqs.union(seqs)

total_seqs = os.listdir(SEQS_DIR)

print(f"{len(processed_seqs)}/{len(total_seqs)}")