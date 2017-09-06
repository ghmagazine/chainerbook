## 前提
- ch6ディレクトリ内にdataディレクトがある
- dataディレクトリの中にCamSeq01というディレクトがある
    - 当該ディレクトリの中に`.png`, `_L.png`, `_L.npz`ファイルがある
    - `_L.npz`は色ファイル(`_L.png`)からchannel1のクラスラベルに変換したファイル
    - クラスラベルへの変換は`color2class.py`で行う

## ファイル読み込み
- `train_image_pointer`, `test_image_pointer`でtrainとtestで使用するファイル名を指定

## 実行
- トレーニング
    - `python train.py`
- テスト
    - `python test.py`

## 結果
- dataディレクトリ内のresultディレクトリに格納されることを想定

## テスト時の設定
- modelバージョン
    - settings.py内のmodel_versionで指定
