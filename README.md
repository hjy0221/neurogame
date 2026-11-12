# NeuroGame

[日本語](https://github.com/hjy0221/neurogame#%E6%97%A5%E6%9C%AC%E8%AA%9E) · [English](https://github.com/hjy0221/neurogame#english) · [한국어](https://github.com/hjy0221/neurogame#%ED%95%9C%EA%B5%AD%EC%96%B4)

## 日本語

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

## English

NeuroGame is an experimental project that challenges biologically inspired neural networks to find the exit of a procedurally generated maze. A recurrent spiking neural network receives wall and exit sensors as input and outputs movement commands.

On the right side of the screen, the training neural network structure: input layer → recurrent middle layer → output layer and the amount of activity in each layer are displayed.

## Included features

- Project structure based on Python 3.12
- About 500 neurons implemented directly in NumPy
- Sparse circular connections including excitatory and inhibitory neurons
- Leaky membrane potential, spikes, refractory period, traces of activity
- Reward-based Hebbian plasticity
- Dense 128 → LSTM 128 recurrent PPO reinforcement learning that uses sensory and spiking activities as input
- 2D Pygame environment for procedural generation focusing on exit navigation
- Entrance/exit DFS maze reflected in collision and wall sensors
- Multiple crossroads and dead ends, single connectable route that can be completed
- Exit arrival compensation and entrance restart loop
- Procedurally generated maze where levels, grids, and dead ends increase with each exit you pass.
- Curriculum that can be learned even in long mazes with exit route checkpoint rewards
-Storable learning weights using exit and wall sensors
- 360-degree 8-way wall distance LiDAR and front, left, and right proximity sensors
- New cell search rewards and repeated visit deductions
- Exit search single reinforcement learning goal
- Input layer → circulation layer → output layer real-time visualization
- Tests run deterministically

## Installation

```bash
cd neurogame
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

In environments without Python 3.12, it may run on lower Python for basic verification, but the official baseline is Python 3.12.

## run

```bash
neurogame
```

Or run `python -m neurogame.app`.

```bash
python -m neurogame.app --neurons 500 --seed 7
python -m neurogame.app --no-plasticity
python -m neurogame.app --no-rl
python -m neurogame.app --headless --steps 300
python -m neurogame.app --headless --steps 5000 --episodes 20 --seed 1 --model NUL --save-model models/forager_exit.npz
python -m neurogame.app --headless --eval --steps 5000 --episodes 10 --seed 10001
python -m neurogame.app --headless --steps 10000 --episodes 10 --seed 1 --model NUL --save-model models/forager_exit.npz
```

## Controls

- `Space`: Pause / Continue
- `R`: Reset environment and brain state
- `P`: Turn plasticity on/off
- `L`: Turn reinforcement learning on/off
- `Esc`: Closed

Training uses multiple environments starting from seed 1, and evaluation is performed starting from seed 10001, which does not overlap. Exploration noise, Actor–Critic updates, and Hebbian plasticity are all disabled in `--eval`. The execution policy does not take BFS path input and uses only wall/exit sensors and sensor-based short-term avoidance memory.

The agent's movements are determined by a neural network, not the keyboard.

## test

```bash
pytest
```

In my current environment I don't have Python 3.12 installed, so I tested with whatever Python was available. Five tests and headless execution passed successfully.

## Next expansion direction

- Replace synthetic cyclic graphs with parts of real connectomes
- Rich visual input based on screen pixels
- Gymnasium environment support
- Save execution records and graphs
- Evolutionary learning or reinforcement learning of sensor-behavior connection

## 한국어

NeuroGame은 생물학적으로 영감을 받은 신경망이 절차 생성 미로의 출구를 찾도록 만든 실험 프로젝트입니다. 순환형 스파이킹 신경망이 벽·출구 센서를 입력으로 받고 이동 명령을 출력합니다.

화면 오른쪽에는 교육용 신경망 구조인 입력층 → 순환 중간층 → 출력층과 각 층의 활동량이 표시됩니다.

## 포함된 기능

- Python 3.12 기준 프로젝트 구조
- NumPy로 직접 구현한 약 500개 뉴런
- 흥분성·억제성 뉴런을 포함한 희소 순환 연결
- 누수 막전위, 스파이크, 불응기, 활동 흔적
- 보상 기반 Hebbian 가소성
- 감각·스파이킹 활동을 입력으로 쓰는 Dense 128 → LSTM 128 recurrent PPO 강화학습
- 출구 탐색에 집중하는 절차 생성 2D Pygame 환경
- 충돌과 벽 센서에 반영되는 입구·출구형 DFS 미로
- 다수의 갈림길과 막다른 길, 완주 가능한 단일 연결 경로
- 출구 도달 보상과 입구 재시작 루프
- 출구를 통과할 때마다 레벨·격자·막다른 길이 증가하는 절차 생성 미로
- 출구 경로 체크포인트 보상으로 긴 미로에서도 학습 가능한 커리큘럼
- 출구와 벽 센서를 사용하는 저장 가능한 학습 가중치
- 360도 8방향 벽 거리 LiDAR와 전·좌·우 근접 센서
- 신규 셀 탐색 보상과 반복 방문 감점
- 출구 탐색 단일 강화학습 목표
- 입력층 → 순환층 → 출력층 실시간 시각화
- 결정론적으로 실행되는 테스트

## 설치

```bash
cd neurogame
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Python 3.12가 없는 환경에서는 기본 확인을 위해 더 낮은 Python에서도 실행될 수 있지만, 공식 기준은 Python 3.12입니다.

## 실행

```bash
neurogame
```

또는 `python -m neurogame.app`을 실행합니다.

```bash
python -m neurogame.app --neurons 500 --seed 7
python -m neurogame.app --no-plasticity
python -m neurogame.app --no-rl
python -m neurogame.app --headless --steps 300
python -m neurogame.app --headless --steps 5000 --episodes 20 --seed 1 --model NUL --save-model models/forager_exit.npz
python -m neurogame.app --headless --eval --steps 5000 --episodes 10 --seed 10001
python -m neurogame.app --headless --steps 10000 --episodes 10 --seed 1 --model NUL --save-model models/forager_exit.npz
```

## 조작법

- `Space`: 일시정지 / 계속하기
- `R`: 환경과 뇌 상태 초기화
- `P`: 가소성 켜기 / 끄기
- `L`: 강화학습 켜기 / 끄기
- `Esc`: 종료

훈련은 시드 1부터 여러 환경을 사용하고, 평가는 겹치지 않는 시드 10001부터 수행합니다. `--eval`에서는 탐험 노이즈, Actor–Critic 업데이트, Hebbian 가소성이 모두 비활성화됩니다. 실행 정책은 BFS 경로를 입력받지 않고 벽·출구 센서와 센서 기반 단기 회피 기억만 사용합니다.

에이전트의 이동은 키보드가 아니라 신경망이 결정합니다.

## 테스트

```bash
pytest
```

현재 환경에서는 Python 3.12가 설치되어 있지 않아 사용 가능한 Python으로 테스트했습니다. 테스트 5개와 헤드리스 실행을 정상 통과했습니다.

## 다음 확장 방향

- 합성 순환 그래프를 실제 커넥톰 일부로 교체
- 화면 픽셀 기반의 풍부한 시각 입력
- Gymnasium 환경 지원
- 실행 기록과 그래프 저장
- 센서·행동 연결의 진화 학습 또는 강화 학습
