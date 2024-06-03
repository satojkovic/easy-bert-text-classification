import argparse
import os
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"], padding=True, truncation=True, max_length=512
    )  # 受け取ったバッチからtextデータを取り出してtokenizerに入れる


def compute_metrics(result):
    labels = result.label_ids
    preds = result.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to dataset (including trian and test csv files)",
    )
    args = parser.parse_args()

    sc_model = BertForSequenceClassification.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=9
    )  # 事前学習済みBERTモデルの読み込み
    sc_model.to(device="mps")  # Apple Silicon
    tokenizer = BertJapaneseTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )  # 事前学習済みのBERTトークナイザーを読み込み

    # CSVデータの読み込み #カラムの指定,#学習データとテストデータは事前に分割済みのためtrainを指定
    train_data = load_dataset(
        "csv",
        data_files=os.path.join(args.dataset_path, "news_train.csv"),
        column_names=["text", "label"],
        split="train",
    )
    # 単語ごとに分割する  #バッチサイズは学習データすべてを指定
    train_data = train_data.map(
        lambda x: tokenize(x, tokenizer), batched=True, batch_size=len(train_data)
    )
    # 学習データのフォーマット指定,Pytorchを指定,input_idsとlabelのカラムを指定
    train_data.set_format("torch", columns=["input_ids", "label"])

    test_data = load_dataset(
        "csv",
        data_files=os.path.join(args.dataset_path, "news_test.csv"),
        column_names=["text", "label"],
        split="train",
    )
    test_data = test_data.map(
        lambda x: tokenize(x, tokenizer), batched=True, batch_size=len(test_data)
    )
    test_data.set_format("torch", columns=["input_ids", "label"])

    # Training
    training_args = TrainingArguments(
        output_dir=os.path.join(args.dataset_path, "results"),
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        warmup_steps=500,  # 学習係数が0からこのステップ数で上昇
        weight_decay=0.01,  # 重みの減衰率
        logging_dir=os.path.join(args.dataset_path, "logs"),
        evaluation_strategy="steps",
    )
    trainer = Trainer(
        model=sc_model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=test_data,
    )
    trainer.train()
    trainer.evaluate()

    # Save model
    sc_model.save_pretrained(args.dataset_path)
    tokenizer.save_pretrained(args.dataset_path)
