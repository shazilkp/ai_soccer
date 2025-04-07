# env.py
import pygame
import numpy as np
import random

WIDTH, HEIGHT = 640, 480
PLAYER_SIZE = 20
BALL_SIZE = 10
PLAYER_SPEED = 5
BALL_SPEED = 6
GOAL_WIDTH = 80

class SoccerEnv:
    def __init__(self, render_mode=True):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT)) if render_mode else None
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode
        self.reset()

    def reset(self):
        self.p1 = pygame.Rect(100, HEIGHT // 2, PLAYER_SIZE, PLAYER_SIZE)
        self.p2 = pygame.Rect(WIDTH - 100, HEIGHT // 2, PLAYER_SIZE, PLAYER_SIZE)
        self.ball = pygame.Rect(WIDTH // 2, HEIGHT // 2, BALL_SIZE, BALL_SIZE)
        self.ball_vel = [0, 0]
        self.done = False
        self.possession = 0  # 0 = none, 1 = p1, 2 = p2
        return self.get_obs()

    def get_obs(self):
        return np.array([
            self.p1.x / WIDTH, self.p1.y / HEIGHT,
            self.p2.x / WIDTH, self.p2.y / HEIGHT,
            self.ball.x / WIDTH, self.ball.y / HEIGHT,
            self.ball_vel[0] / BALL_SPEED, self.ball_vel[1] / BALL_SPEED,
            self.possession / 2
        ], dtype=np.float32)

    def step1(self, action1, action2):
        self._move_player(self.p1, action1)
        self._move_player(self.p2, action2)
        self._handle_possession()
        self._move_ball()
        reward = 0

        if self.ball.left <= 0:
            reward = -1
            self.done = True
        elif self.ball.right >= WIDTH:
            reward = 1
            self.done = True

        if self.render_mode:
            self.render()

        return self.get_obs(), reward, self.done, {}




    def step(self, action1, action2):
        old_dist = self._distance(self.p1.center, self.ball.center)
        #print(action1);
        self._move_player(self.p1, action1)
        self._move_player(self.p2, action2)
        self._handle_possession()

        reward = 0.0

        # Reward for gaining possession
        if self.possession == 1:
            reward += 0.01

        # Reward for moving closer to the ball
        new_dist = self._distance(self.p1.center, self.ball.center)
        if new_dist < old_dist:
            reward += 0.02
        
        if action1 == 4 and self.possession != 1:
            reward -= 0.1  # Penalty for kicking without possession


        # Optional: detect kicking (if action1 == 4 and possession)
        if action1 == 4 and self.possession == 1:
           # print("kicking logic reached")
            self.try_kick(self.p1, [1, 0])  # Kick right
            # Add extra reward if near the opponent's goal
            if self.ball.x > WIDTH * 0.7:
                reward += 0.7  # More incentive to kick near goal
            else:
                reward += 0.05  # Base reward for trying to kick

        self._move_ball()

        goal_top = HEIGHT // 2 - GOAL_WIDTH // 2
        goal_bottom = HEIGHT // 2 + GOAL_WIDTH // 2

        # Check for own goal (ball crosses left boundary *and* is within goal height)
        if self.ball.left <= 0 and goal_top <= self.ball.y <= goal_bottom:
            print("Own goal")
            reward -= 0.5
            self.done = True

        # Check for scoring (ball crosses right boundary *and* is within goal height)
        elif self.ball.right >= WIDTH and goal_top <= self.ball.y <= goal_bottom:
            print("Scored a goal!")
            reward += 2.0
            if action1 == 4:  # last action was a kick
                reward += 0.5 
            self.done = True


        if self.render_mode:
            self.render()

        return self.get_obs(), reward, self.done, {}
        
    def _distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def try_kick(self, player, direction):
        if self.possession == 1 and player == self.p1:
            self.ball_vel = [direction[0] * BALL_SPEED, direction[1] * BALL_SPEED]
            self.possession = 0
        elif self.possession == 2 and player == self.p2:
            self.ball_vel = [direction[0] * BALL_SPEED, direction[1] * BALL_SPEED]
            self.possession = 0

    def _move_player(self, player, action):
        if action == 0: player.y -= PLAYER_SPEED  # up
        elif action == 1: player.y += PLAYER_SPEED  # down
        elif action == 2: player.x -= PLAYER_SPEED  # left
        elif action == 3: player.x += PLAYER_SPEED  # right
        elif action == 4: pass  # kick handled externally

        player.x = max(0, min(WIDTH - PLAYER_SIZE, player.x))
        player.y = max(0, min(HEIGHT - PLAYER_SIZE, player.y))

    def _handle_possession(self):
        if self.ball_vel == [0, 0]:  # Only allow possession if the ball is stationary
            if self.ball.colliderect(self.p1):
                self.possession = 1
                self.ball.center = self.p1.center
            elif self.ball.colliderect(self.p2):
                self.possession = 2
                self.ball.center = self.p2.center


    def _move_ball(self):
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        if self.ball.top <= 0 or self.ball.bottom >= HEIGHT:
            self.ball_vel[1] *= -1

    def render(self):
        self.screen.fill((34, 139, 34))
        pygame.draw.rect(self.screen, (255, 255, 255), (0, HEIGHT//2 - GOAL_WIDTH//2, 10, GOAL_WIDTH))
        pygame.draw.rect(self.screen, (255, 255, 255), (WIDTH-10, HEIGHT//2 - GOAL_WIDTH//2, 10, GOAL_WIDTH))
        pygame.draw.rect(self.screen, (0, 0, 255), self.p1)
        pygame.draw.rect(self.screen, (255, 0, 0), self.p2)
        pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)
        pygame.display.flip()
        self.clock.tick(60)

