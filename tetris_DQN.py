
import numpy as np
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import pygame
import os

# === Tetris Environment ===
class TetrisEnv:
    def __init__(self, rows=20, cols=10, block_size=30):
        self.rows = rows
        self.cols = cols
        self.block_size = block_size
        self.width = cols * block_size
        self.height = rows * block_size
        self.tetromino_shapes = [
            [[1, 1, 1, 1]], [[2, 0, 0], [2, 2, 2]],
            [[0, 0, 3], [3, 3, 3]], [[4, 4], [4, 4]],
            [[0, 5, 5], [5, 5, 0]], [[0, 6, 0], [6, 6, 6]],
            [[7, 7, 0], [0, 7, 7]]
        ]
        self.colors = [(0, 0, 0), (0, 255, 255), (0, 0, 255), (255, 165, 0),
                       (255, 255, 0), (0, 255, 0), (160, 32, 240), (255, 0, 0)]
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.spawn_block()
        self.done = False
        self.prev_lines_cleared = 0
        return self.get_state()

    def spawn_block(self):
        self.current_shape = random.choice(self.tetromino_shapes)
        self.current_x = self.cols // 2 - len(self.current_shape[0]) // 2
        self.current_y = 0

    def check_collision(self, shape, x, y):
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    nx, ny = x + j, y + i
                    if nx < 0 or nx >= self.cols or ny >= self.rows or (ny >= 0 and self.board[ny][nx]):
                        return True
        return False

    def merge(self, shape, x, y):
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    self.board[y + i][x + j] = cell

    def clear_lines(self):
        new_board = [row for row in self.board if any(cell == 0 for cell in row)]
        lines_cleared = self.rows - len(new_board)
        while len(new_board) < self.rows:
            new_board.insert(0, [0] * self.cols)
        self.board = np.array(new_board)
        return lines_cleared

    def get_state(self):
        flat_board = self.board.flatten()
        block_info = np.array([len(self.current_shape), len(self.current_shape[0]), self.current_x])
        return np.concatenate((flat_board, block_info))

    def step(self, action):
        rotation_count, target_x = action
        shape = self.current_shape
        for _ in range(rotation_count % 4):
            shape = list(zip(*shape[::-1]))
        x = min(target_x, self.cols - len(shape[0]))
        y = 0
        while not self.check_collision(shape, x, y):
            y += 1
        y -= 1
        if y < 0:
            self.done = True
            self.prev_lines_cleared = 0
            return self.get_state(), -10, self.done

        self.merge(shape, x, y)
        lines_cleared = self.clear_lines()
        self.prev_lines_cleared = lines_cleared
        self.spawn_block()
        if self.check_collision(self.current_shape, self.current_x, self.current_y):
            self.done = True
        return self.get_state(), 0, self.done

    def count_holes(self):
        holes = 0
        for x in range(self.cols):
            block_found = False
            for y in range(self.rows):
                if self.board[y][x]:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def get_height(self):
        heights = [0] * self.cols
        for x in range(self.cols):
            for y in range(self.rows):
                if self.board[y][x]:
                    heights[x] = self.rows - y
                    break
        return sum(heights) // self.cols

    def get_action_space(self):
        return [(r, x) for r in range(4) for x in range(self.cols)]

    def render(self):
        self.screen.fill((0, 0, 0))
        for y in range(self.rows):
            for x in range(self.cols):
                color = self.colors[self.board[y][x]]
                pygame.draw.rect(self.screen, color,
                    (x * self.block_size, y * self.block_size, self.block_size, self.block_size))
        pygame.display.flip()
        self.clock.tick(30)

# === Reward Function ===
def compute_reward(env, x, shape, lines_cleared, died):
    holes = env.count_holes()
    height = env.get_height()
    center_x = env.cols // 2
    distance_from_center = abs(x - center_x)

    # 🔥 줄 제거에 따른 강한 보상
    if lines_cleared == 1:
        line_score = 1000
    elif lines_cleared == 2:
        line_score = 2500
    elif lines_cleared == 3:
        line_score = 4000
    elif lines_cleared == 4:
        line_score = 8000
    else:
        line_score = -200  # 줄 못 없앴으면 약한 패널티

    # 👇 보조적인 상태 보상
    survival_bonus = 200
    height_penalty = -height * 5
    holes_penalty = -holes * 50
    center_penalty = -distance_from_center * 2
    death_penalty = -1000 if died else 0

    reward = (
        line_score
        + survival_bonus
        + height_penalty
        + holes_penalty
        + center_penalty
        + death_penalty
    )
    return reward

def warmup_with_heuristic(buffer, env, weights, action_space, samples=3000):
    def get_column_heights(board):
        return [next((env.rows - y for y in range(env.rows) if board[y][x]), 0) for x in range(env.cols)]

    def get_bumpiness(heights):
        return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

    for _ in range(samples):
        state = env.reset()
        while not env.done:
            best_score = float('-inf')
            best_action = None

            for idx, (rot, x) in enumerate(action_space):
                shape = env.current_shape
                for _ in range(rot):
                    shape = list(zip(*shape[::-1]))

                x = min(x, env.cols - len(shape[0]))
                y = 0
                while not env.check_collision(shape, x, y):
                    y += 1
                y -= 1
                if y < 0:
                    continue

                temp_board = copy.deepcopy(env.board)
                env.merge(shape, x, y)
                lines = env.clear_lines()
                score = (
                    5.0 * lines
                    - 1.0 * env.count_holes()
                    - 0.5 * env.get_height()
                    - 0.3 * get_bumpiness(get_column_heights(env.board))
                )
                env.board = temp_board

                if score > best_score:
                    best_score = score
                    best_action = idx

            if best_action is not None:
                action = action_space[best_action]
                next_state, _, done = env.step(action)
                reward = 100 if env.prev_lines_cleared > 0 else -10
                buffer.push(state, best_action, reward, next_state, done)
                state = next_state

# === DQN 클래스 ===
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# === 리플레이 버퍼 ===
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return zip(*samples)

    def __len__(self):
        return len(self.buffer)

# === 학습 루프 ===
def train_dqn():
    env = TetrisEnv()
    action_space = env.get_action_space()
    state_dim = len(env.get_state())
    action_dim = len(action_space)

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)

    if os.path.exists("best_dqn.pth"):
        print("📦 기존 best_dqn.pth 로드")
        policy_net.load_state_dict(torch.load("best_dqn.pth"))
        target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    if os.path.exists("best_weights.npy"):
        weights = np.load("best_weights.npy")
        print("💡 GA 기반 best_weights 로딩 및 warm-up 중...")
        warmup_with_heuristic(buffer, env, weights, action_space, samples=3000)

    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    min_epsilon = 0.1
    max_score = -float('inf')
    if os.path.exists("best_score.txt"):
        with open("best_score.txt") as f:
            max_score = float(f.read())

    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
            if random.random() < epsilon:
                action_idx = random.randint(0, len(action_space) - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action_idx = policy_net(state_tensor).argmax().item()
            action = action_space[action_idx]
            next_state, _, done = env.step(action)
            reward = compute_reward(env, action[1], env.current_shape, env.prev_lines_cleared, done)
            buffer.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            if episode % 10 == 0:
                env.render()
            if done:
                break
            if len(buffer) >= 64:
                states, actions, rewards, next_states, dones = buffer.sample(64)
                states = torch.tensor(np.array(list(states)), dtype=torch.float32)
                actions = torch.tensor(list(actions)).unsqueeze(1)
                rewards = torch.tensor(list(rewards), dtype=torch.float32)
                next_states = torch.tensor(np.array(list(next_states)), dtype=torch.float32)
                dones = torch.tensor(list(dones), dtype=torch.float32)

                q_values = policy_net(states).gather(1, actions).squeeze()
                next_actions = policy_net(next_states).argmax(1).unsqueeze(1)
                next_q = target_net(next_states).gather(1, next_actions).squeeze()
                targets = rewards + gamma * next_q * (1 - dones)

                loss = F.mse_loss(q_values, targets)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        if total_reward > max_score:
            max_score = total_reward
            torch.save(policy_net.state_dict(), "best_dqn.pth")
            with open("best_score.txt", "w") as f:
                f.write(str(max_score))
            print(f"🎉 New High Score: {max_score:.2f} (Model Saved)")
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode} - Score: {total_reward} - Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    train_dqn()
