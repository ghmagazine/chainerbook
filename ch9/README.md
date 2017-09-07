# dqn_agent.py

9章で紹介したDQNのサンプルコードです。

* dqn_agent.py: サンプルコード本体
* network.py: ネットワーク構造を定義したPythonファイル。dqn_agent.pyでimportされる

## 必要なパッケージ

* Chainer
* OpenAI gym
* NumPy

## 実行方法

    python ./dqn_agent.py

実行するとウィンドウが開きCartPoleがどのように動いているのかが確認できます。標準出力に各エピソードで得た報酬を出力します。

199行めの変数`render`で、ウィンドウを開きアニメーションを表示するか否かを変えることができます。`render=False`でウィンドウとアニメーションの表示を行わないようにできます。

network.pyには3つのネットワーク `MLP3DQNet`、`MLP3DQNet`と`MLP3DQNet`が定義してあります。書籍で紹介している3、2、1層のネットワークに対応しています。この実験を再現するためには205行目で指定している`DQNGymAgent`クラスのイニシャライズで指定している引数を変更します。例えば`model_network=network.MLP3DQNet`->`model_network=network.MLP1DQNet`のようにします(この例は1層ネットワークを使用するように変更する場合です)。