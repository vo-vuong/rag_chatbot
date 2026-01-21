"""
Create a testing file from qr_smartphone_dataset.json to qr_smartphone_dataset.xlsx

Usage:
    conda activate rag_chatbot && python rag_evaluation/prepare_testing_data/create_testing_file.py
"""

import json
from pathlib import Path

import pandas as pd


def main():
    script_dir = Path(__file__).parent
    json_path = script_dir / "qr_smartphone_dataset.json"
    output_path = script_dir / "qr_smartphone_dataset.xlsx"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)

    print(f"File {output_path.name} has been created with {len(df)} rows.")


if __name__ == "__main__":
    main()
