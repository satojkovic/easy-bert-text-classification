import glob
import os
import torch
import random
import argparse

from transformers import (
    AutoConfig,
    BertForSequenceClassification,
    BertJapaneseTokenizer,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir_path", required=True, help="Path to model directory"
    )
    args = parser.parse_args()

    # Load fine tuned model
    config = AutoConfig.from_pretrained(
        os.path.join(args.model_dir_path, "config.json")
    )
    model = BertForSequenceClassification(config)
    loaded_model = model.from_pretrained(args.model_dir_path).to(device="mps")

    # Load fine tuned tokenizer
    tokenizer = BertJapaneseTokenizer(os.path.join(args.model_dir_path, "vocab.txt"))
    loaded_tokenizer = tokenizer.from_pretrained(args.model_dir_path)

    category_list = [
        "dokujo-tsushin",
        "it-life-hack",
        "livedoor-homme",
        "kaden-channel",
        "movie-enter",
        "sports-watch",
        "smax",
        "topic-news",
        "peachy",
    ]
    category = random.choice(category_list)
    print("GT category:", category)
    random_number = random.randrange(1, 99)
    sample_path = os.path.join(args.model_dir_path, "text")
    files = glob.glob(os.path.join(sample_path, category, "*.txt"))
    file = files[random_number]
    file_name = os.path.basename(file)

    dir_files = os.listdir(path=sample_path)
    dirs = [f for f in dir_files if os.path.isdir(os.path.join(sample_path, f))]

    with open(file, "r") as f:
        sample_text = f.readlines()[3:]
        sample_text = "".join(sample_text)
        sample_text = sample_text.translate(
            str.maketrans({"\n": "", "\t": "", "\r": "", "\u3000": ""})
        )

    print(sample_text)

    max_length = 512
    words = loaded_tokenizer.tokenize(sample_text)
    word_ids = loaded_tokenizer.convert_tokens_to_ids(words)
    word_tensor = torch.tensor(
        [word_ids[:max_length]]
    )  # テンソルに変換　スライスを使って単語が512より多い場合は切る
    x = word_tensor.to(device="mps")
    y = loaded_model(x)
    pred = y[0].argmax(-1)
    print("predict-result:", dirs[pred])
