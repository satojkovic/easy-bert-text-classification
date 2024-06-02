import glob  # ファイルの取得に使用
import os
import argparse
import csv
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to data directory (text directory path).",
    )
    parser.add_argument("--output_path", required=True, help="Output path")
    args = parser.parse_args()

    path = args.data_path
    dir_files = os.listdir(
        path=path
    )  # 指定したpathのファイルとディレクトリの一覧を取得
    dirs = [
        f for f in dir_files if os.path.isdir(os.path.join(path, f))
    ]  # ディレクトリをリストとして取り出す

    text_label_data = []  # 文章とラベルのセット
    dir_count = 0  # ディレクトリ数のカウント
    file_count = 0  # ファイル数のカウント
    for i, directory in enumerate(dirs):  # ディレクトリの数だけループ処理
        files = glob.glob(os.path.join(path, directory, "*.txt"))  # ファイルの一覧
        dir_count += 1
        for file in files:
            if (
                os.path.basename(file) == "LICENSE.txt"
            ):  # LICENSE.txtは除外する（ループをスキップ）
                continue
            with open(file, "r") as f:  # ファイルを開く
                text = f.readlines()[3:]  # 指定の行だけ読み込む 4行目以降を読み込む
                text = "".join(
                    text
                )  # リストなのでjoinで結合する　空の文字列に結合して一つの文字列にする
                text = text.translate(
                    str.maketrans({"\n": "", "\t": "", "\r": "", "\u3000": ""})
                )  # 不要な文字を除去する
                text_label_data.append(
                    [text, i]
                )  # 本文とディレクトリ番号をリストに加える
            file_count += 1
            print(
                "\rfiles: " + str(file_count) + " " + "dirs: " + str(dir_count), end=""
            )

    # Create dataset
    news_train, news_test = train_test_split(text_label_data, shuffle=True)
    news_path = args.output_path
    with open(os.path.join(news_path, "news_train.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(news_train)
    with open(os.path.join(news_path, "news_test.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(news_test)
