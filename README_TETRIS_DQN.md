Tetris AI with Deep Q-Network (DQN)

본 프로젝트는 강화학습(Deep Q-Learning)을 통해 고전 게임 "Tetris"를 학습하고 최적의 플레이를 수행하도록 하는 AI 모델을 구현한 것입니다. 단순한 DQN에서 시작해 휴리스틱 기반 warm-up과 보상 함수 개선을 반복하며 모델 성능을 높였습니다.

🚀 프로젝트 개요

목표: 테트리스 게임에서 가능한 한 오랫동안 생존하며 줄을 많이 제거하는 AI 에이전트 학습

환경: PyTorch, Pygame, Numpy

학습 알고리즘: DQN (Double DQN 구조 일부 적용), Heuristic-based warm-up

🧑‍💻 주요 구성 요소

1. TetrisEnv (환경 클래스)

줄 제거, 충돌 체크, 블록 병합 등의 테트리스 규칙을 구현

get_state() 함수로 flatten된 보드 + 블록 정보 상태 반환

step()에서 action에 따라 블록 위치 결정 및 보상 계산 트리거

2. compute_reward (보상 함수)

줄 제거를 중심으로 다양한 요소 반영:

줄 제거 수에 따른 보상 (1~4줄 제거 시 높은 점수 부여)

구멍 수, 전체 높이, bumpiness(울퉁불퉁함) 등은 페널티 부여

하단 채움 유도와 깊은 낙하를 장려하는 보조 보상

줄 제거 실패나 게임 오버 시 강한 페널티 부여

if lines_cleared == 1:
    line_score = 800
elif lines_cleared == 2:
    line_score = 1400
...
reward = line_score + fill_score * 0.5 - holes * 30 - height * 1.5 ...

3. warmup_with_heuristic

GA로 학습된 휴리스틱 가중치를 사용해 buffer를 사전 채움

보드 시뮬레이션을 통해 최적 action을 평가하고 push

초기 DQN이 무작위 정책 대신 어느 정도 학습된 상태에서 시작하도록 유도

4. DQN 구조

self.fc1 = nn.Linear(input_dim, 256)
self.fc2 = nn.Linear(256, 128)
self.out = nn.Linear(128, output_dim)

💪 훈련 전략 요약

Epsilon-greedy 탐험 방식 사용 (1.0 → 0.05)

buffer size: 10,000 / batch size: 64

보상 기반 모델 저장 조건: 최고 점수 갱신 시

10 episode마다 target network 동기화

📊 실험 결과 및 분석

✅ 주요 개선 사항

줄 제거 중심 보상 구조로 전환하면서 중앙 정렬, 단순한 생존 중심 전략에서 벗어남

fill_score, row_fill_reward, drop_depth 등으로 줄 완성 유도 강화

warm-up 단계 추가로 초기 학습 속도 향상 및 안정성 확보

❌ 문제점

줄 제거보다도 빠르게 사망하는 사례 존재

여전히 오른쪽 또는 중앙 쏠림 현상이 후반부에 나타남

⚡ 향후 개선 방안

생존 시간에 대한 보상 항목 추가: 오래 살수록 점수 증가

Dueling DQN, PER 등 고급 기법 적용

더 정교한 줄 완성 감지 및 예측 로직 적용

📑 실행 방법

Python 가상환경 설정 및 패키지 설치:

conda create -n tetris python=3.10
conda activate tetris
pip install torch pygame numpy

모델 파일 준비 (선택):

best_weights.npy : GA 기반 휴리스틱 가중치

best_dqn.pth : 이전 학습된 DQN

실행:

python t.py

📄 관련 파일 구성

.
├── t.py                    # 메인 실행 코드
├── best_dqn.pth           # (선택) 저장된 DQN 모델 가중치
├── best_weights.npy       # (선택) 휴리스틱 warmup용 가중치
├── best_score.txt         # 최고 점수 저장 파일
├── README.md              # 현재 문서

😊 저자 코멘트

이 프로젝트는 강화학습 초심자에게도 실험적으로 즐기기 좋은 예시입니다. 실시간 렌더링으로 눈에 보이는 결과를 확인하면서 DQN이 어떻게 학습되는지를 체감할 수 있도록 설계했습니다.

궁극적으로는 줄 제거와 생존 사이에서 균형 잡힌 의사결정을 내리는 AI를 목표로 하고 있으며, 여전히 개선의 여지는 많습니다. 많은 피드백 환영합니다!