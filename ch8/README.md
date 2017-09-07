# ram_network.py

8章で紹介しているRAMのソースコードです。

## 必要なパッケージ

* Chainer
* scikit-learn: データセットMNISTをダウンロードするために使用します
* numpy
* PIL (Pillow)

## 実行方法

    python ram_netwok.py

実行するとMNISTデータセットをダウンロードし、RAMの学習を行います。学習途中でロスの値と精度を標準出力に出力します。ソースコード内で指定された10エポックを終了すると、学習済みのモデルを`saved_model`というファイル名で保存します。なおscikit-learnのデフォルトではダウンロードしたデータセットは`~/scikit_learn_data/`以下の保存されます。

