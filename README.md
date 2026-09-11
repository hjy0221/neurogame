# NeuroGame

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
