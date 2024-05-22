import os
import json

log_dir = "logs/3558f1342/render_log"

def collect_logs(log_dir: str):
    result = {
        "succeeded": [],
        "failed": []
    }
    for log_file in os.listdir(log_dir):
        if not log_file.endswith(".log"):
            continue
        with open(os.path.join(log_dir, log_file), "r") as f:
            single_result = json.load(f)
        result["succeeded"].extend(list(set(single_result["succeeded"])))
        result["failed"].extend(list(set(single_result["failed"])))
    return result

result = collect_logs(log_dir)

with open("multi_render.log.json", "w") as f:
    json.dump(result, f)

with open("filter", "w") as f:
    f.write("\n".join(result["succeeded"]))