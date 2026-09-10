# NeuroForager

Swift + SwiftUI Canvas 기반 iOS 17+ 온디바이스 신경망 탐색 시뮬레이션 v0.1.

## 실행

`NeuroForager.xcodeproj`를 Xcode 16 이상에서 열고 NeuroForager 스킴과 iPhone 시뮬레이터를 선택해 실행합니다. 실제 기기에서는 Signing & Capabilities에서 본인의 Team을 선택하세요. 외부 패키지나 AI API 키는 필요하지 않습니다.

```sh
xcodebuild -project NeuroForager.xcodeproj -scheme NeuroForager -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build
```

## 구성

- Environment: 이동, 경계 충돌, 먹이 소비와 재생성.
- SensorEncoder: 먹이 방향 16채널과 내부 상태를 포함한 32개 입력.
- BrainSimulator: 500개 인공 뉴런, 4,000개 희소 재귀 연결, 막전위 누설, 적응 발화 문턱, 발화 흔적. 순수 Swift 계산.
- ActionDecoder: 네 운동 출력 집단에서 회전과 추진력 계산.
- Visualization: 세계와 신경망을 Canvas로 표시. 실제 재귀 연결선, 발화 노드, 흥분성/억제성 색상, 노드 선택과 확대 화면.

## 모델 범위와 확장

생물학적 영감을 받은 인공 모델이며 실제 초파리 connectome 재현은 아닙니다. 학습된 정책이나 온라인 학습은 없으며 먹이 방향에 대한 선천적 입력 배선을 사용합니다. 먹이 수집 성능은 실험적입니다. 에너지 0에서도 시뮬레이션은 계속됩니다.

`BrainSimulator.buildNetwork()`의 생성 배선을 데이터 로더로 교체하면 실제 연결 데이터를 도입할 수 있습니다. `NeuralConnection`은 렌더러가 소비하는 읽기 전용 연결 표현입니다. 향후 world model은 센서 인코더와 뇌 사이에서 예측 입력을 제공하는 별도 모듈로 추가할 수 있습니다. 시각화 좌표는 고정된 추상 배치이며 해부학적 위치를 뜻하지 않습니다.

현재 렌더링과 시뮬레이션은 메인 스레드에서 수행됩니다. 더 큰 네트워크로 확장할 때는 시뮬레이션 actor와 스냅샷 게시 주기를 분리하세요.
