# NeuroGame

[日本語](README.ja.md) · [English](README.md) · [한국어](README.ko.md)

NeuroGameは、生物学的にインスピレーションを受けたニューラルネットワークが手続き生成迷路の出口を見つけるように作った実験プロジェクトです。循環型スパイキングニューラルネットワークが壁・出口センサを入力として受け取り、移動指令を出力します。

画面右側には、教育用ニューラルネットワーク構造である入力層→循環中間層→出力層と各層の活動量が表示されます。

## 埋め込み機能

- Python 3.12ベースのプロジェクト構造
- NumPyで直接実装した約500個のニューロン
- 興奮性・抑制性ニューロンを含む希少循環接続
- 漏水膜電位、スパイク、不応器、活動の兆候
- 報酬ベースのHebbian可塑性
- 感覚・スパイキング活動を入力で使うDense 128 → LSTM 128 recurrent PPO強化学習
- 出口ナビゲーションに集中するプロシージャ生成2D Pygame環境
- 衝突と壁センサーに反映される入口・出口型DFS迷路
- 複数の分岐路と行き止まりの道、完成可能な単一の接続経路
- 出口到達報酬と入口再開ループ
- 出口を通過するたびにレベル・格子・幕の長さ増加する手続き生成迷路
- 出口経路チェックポイント補償で長い迷路でも学習可能なカリキュラム
- 出口と壁センサーを使用した保存可能な学習重み
- 360度8方向壁距離LiDARと前・左・右近接センサー
- 新規セルナビゲーション報酬と繰り返し訪問減点
- 出口探索単一強化学習目標
- 入力層→循環層→出力層リアルタイム可視化
- 決定論的に実行されるテスト

## インストール

```bash
cd neurogame
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Python 3.12がない環境では、基本的な検証のために低いPythonでも実行できますが、公式の基準はPython 3.12です。

## 実行

```bash
neurogame
```

または`python -m neurogame.app`を実行します。

```bash
python -m neurogame.app --neurons 500 --seed 7
python -m neurogame.app --no-plasticity
python -m neurogame.app --no-rl
python -m neurogame.app --headless --steps 300
python -m neurogame.app --headless --steps 5000 --episodes 20 --seed 1 --model NUL --save-model models/forager_exit.npz
python -m neurogame.app --headless --eval --steps 5000 --episodes 10 --seed 10001
python -m neurogame.app --headless --steps 10000 --episodes 10 --seed 1 --model NUL --save-model models/forager_exit.npz
```

## 操作法

- `Space`：一時停止/続行
- `R`：環境と脳状態の初期化
- `P`：可塑性のオン/オフ
- `L`：強化学習のオン/オフ
- `Esc`: 終了

訓練はシード1から複数の環境を使用し、評価は重複しないシード10001から行われます。 `--eval`では、探索ノイズ、Actor-Criticアップデート、Hebbian可塑性がすべて無効になります。実行ポリシーはBFS経路を入力せず、壁・出口センサーとセンサーベースの短期回避記憶のみを使用します。

エージェントの移動はキーボードではなくニューラルネットワークによって決定されます。

## テスト

```bash
pytest
```

現在の環境では、Python 3.12がインストールされておらず、利用可能なPythonでテストしました。テスト5つとヘッドレス実行を正常に通過しました。

## 次の拡張方向

- 合成循環グラフを実際のコネクトムの一部に置き換える
- 画面ピクセルベースの豊富な視覚入力
- Gymnasium環境サポート
- 実行履歴とグラフを保存
- センサー・行動接続の進化学習または強化学習
