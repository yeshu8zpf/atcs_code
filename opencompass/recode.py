import csv, json, os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default="summary/summary_20260107_000312.csv")
parser.add_argument('--recode_file', type=str, default="recode.json")
args = parser.parse_args()
csv_file = args.file

scores = []

with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)  
    
    for row in reader:
        if not row:
            continue
        try:
            score = float(row[-1])
        except ValueError:
            print(f"[WARNING] Skip non-numeric score row: {row}")
            continue
        scores.append(score)

if not scores:
    print(f"[WARNING] No valid numeric scores found in {csv_file}, skip recoding.")
    raise SystemExit(0)

avg = sum(scores) / len(scores)

print(f"{len(scores)} tasks")
print(f"average score = {avg:.4f}")

if os.path.exists(args.recode_file):
    with open(args.recode_file, 'r') as f:
        recode_dict = json.load(f)
else:
    recode_dict = {}

recode_dict["  ".join(csv_file.split('/')[1:-3])] = avg

with open(args.recode_file, 'w') as f:
    json.dump(recode_dict, f, indent=4)
