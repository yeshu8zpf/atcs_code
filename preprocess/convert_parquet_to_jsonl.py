import os
import json
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="data")
parser.add_argument('--input_file', type=str, default="tulu-3-sft-mixture")
parser.add_argument('--output_file', type=str, default='tulu3.jsonl')
args = parser.parse_args()

input_dir = f"{args.dir}/{args.input_fiel}"   
output_file = f"{args.dir}/{args.output_file}"

def normalize(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value

with open(output_file, "w", encoding="utf-8") as fout:
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".parquet"):
            print(f"Converting {fname} ...")
            table = pq.read_table(os.path.join(input_dir, fname))
            df = table.to_pandas()

            for record in tqdm(df.to_dict(orient="records")):
                record = {k: normalize(v) for k, v in record.items()}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

print("completed, output file:", output_file)

