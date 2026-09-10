# NeuroGame

NeuroGame은 생물학적으로 영감을 받은 신경망이 작은 2D 게임 안에서 먹이를 찾도록 만든 v0.1 실험 프로젝트입니다. 순환형 스파이킹 신경망이 시각·거리 센서를 입력으로 받고 이동 명령을 출력합니다.

화면 오른쪽에는 교육용 신경망 구조인 입력층 → 순환 중간층 → 출력층과 각 층의 활동량이 표시됩니다.

## 포함된 기능

- Python 3.12 기준 프로젝트 구조
- NumPy로 직접 구현한 약 500개 뉴런
- 흥분성·억제성 뉴런을 포함한 희소 순환 연결
- 누수 막전위, 스파이크, 불응기, 활동 흔적
- 보상 기반 Hebbian 가소성
- 먹이를 찾는 2D Pygame 환경
- 방향별 먹이 감지와 벽·거리 센서
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
python -m neurogame.app --headless --steps 300
```

## 조작법

- `Space`: 일시정지 / 계속하기
- `R`: 환경과 뇌 상태 초기화
- `P`: 가소성 켜기 / 끄기
- `Esc`: 종료

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
