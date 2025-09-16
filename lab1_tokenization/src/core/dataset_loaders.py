def load_raw_text_data(dataset_path: str) -> str:
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text