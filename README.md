# 🎮 Tetris AI with Deep Q-Network (DQN)

본 프로젝트는 **강화학습(Deep Q-Learning)** 기반으로 고전 게임 **Tetris**를 학습하고, 최적의 플레이를 수행하도록 하는 AI 모델을 구현한 예제입니다.
단순한 DQN에서 시작해 **휴리스틱 기반 warm-up**과 **보상 함수 개선**을 반복하면서 성능을 향상시켰습니다.

---

## 🚀 프로젝트 개요

* **목표**: 테트리스에서 오래 생존하며 줄을 최대한 많이 제거하는 AI 학습
* **사용 환경**: `Python`, `PyTorch`, `Pygame`, `NumPy`
* **핵심 기법**:

  * DQN (Double DQN 일부 적용)
  * GA 기반 Heuristic warm-up

---

## 🧩 주요 구성 요소

### 1. `TetrisEnv`: 환경 클래스

* 줄 제거, 충돌, 블록 병합 등의 테트리스 규칙 구현
* `get_state()`:

  * Flatten된 보드 상태 + 블록 정보를 반환
* `step()`:

  * 선택한 action에 따라 블록 위치 결정
  * 보상 계산 및 게임 진행

---

### 2. `compute_reward`: 보상 함수

```python
if lines_cleared == 1:
    line_score = 800
elif lines_cleared == 2:
    line_score = 1400
...
reward = line_score + fill_score * 0.5 - holes * 30 - height * 1.5 ...
```

**구성 요소**:

* ✅ **보상**: 줄 제거 수 (1\~4줄) 에 비례하여 점수 부여
* ❌ **패널티**:

  * 구멍 수 (`holes`)
  * 전체 높이 (`height`)
  * 울퉁불퉁함 (`bumpiness`)
* 🔄 **보조 보상**:

  * 하단 채움 유도 (`fill_score`)
  * 깊은 낙하 (`drop_depth`)
* 💥 **게임 오버 또는 줄 제거 실패 시** 큰 패널티 부여

---

### 3. `warmup_with_heuristic`: 휴리스틱 기반 워밍업

* GA로 학습된 가중치를 이용해 buffer를 사전 채움
* 보드 시뮬레이션 기반으로 최적의 action 선택
* 무작위 정책 대신 어느 정도 학습된 상태에서 학습 시작

---

### 4. DQN 구조

```python
self.fc1 = nn.Linear(input_dim, 256)
self.fc2 = nn.Linear(256, 128)
self.out = nn.Linear(128, output_dim)
```

* 3층 fully-connected 구조
* MSE Loss, Epsilon-Greedy 탐색 정책 사용

---

## 💪 훈련 전략 요약

* 탐색 정책: Epsilon-Greedy (초기 1.0 → 최종 0.05)
* Replay Buffer: 10,000개
* Batch Size: 64
* Target Network: 10 episode마다 동기화
* 모델 저장: 최고 점수 갱신 시 `best_dqn.pth`로 저장

---

## 📊 실험 결과 및 분석

### ✅ 주요 개선

* 중앙 정렬과 생존 위주의 단순 전략 → **줄 제거 중심 전략으로 전환**
* 보조 보상(`fill_score`, `drop_depth`) → **줄 완성 유도**
* warm-up 도입으로 초기 학습 속도 및 안정성 확보

### ❌ 발견된 문제

* 초반 줄 제거보다 **빠른 사망** 빈도 존재
* 후반부에 **중앙 또는 오른쪽으로 쏠리는 현상**

---

## ⚡ 향후 개선 방안

* 생존 시간 기반 보상 추가
* `Dueling DQN`, `Prioritized Experience Replay (PER)` 도입
* 더 정교한 줄 완성 예측 로직 설계

---

## 🛠 실행 방법

### 1. 가상환경 설정

```bash
conda create -n tetris python=3.10
conda activate tetris
pip install torch pygame numpy
```

### 2. 모델 파일 준비 (선택)

* `best_weights.npy`: 휴리스틱 warm-up용 GA 가중치
* `best_dqn.pth`: 사전 학습된 DQN 가중치

### 3. 실행

```bash
python tetris_DQN.py
```

---

## 📁 프로젝트 파일 구조

```
.
├── tetris_DQN.py           # 메인 실행 코드
├── best_dqn.pth            # (선택) 저장된 DQN 모델 가중치
├── best_weights.npy        # (선택) 휴리스틱 warmup용 가중치
├── best_score.txt          # 최고 점수 저장 파일
├── README.md               # 현재 문서
```

---

## 😊 저자 코멘트

이 프로젝트는 **강화학습 입문자에게 적합한 실험용 예제**로,
실시간 렌더링을 통해 **DQN이 학습하는 과정을 눈으로 확인**할 수 있도록 구성했습니다.

궁극적으로는 **줄 제거와 생존 사이에서 균형 잡힌 의사결정**을 내리는 AI를 목표로 하며,
여전히 개선 여지가 많은 만큼 **여러분의 피드백을 환영**합니다!
