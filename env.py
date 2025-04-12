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

    def load_assets(self):
        self.field_image = pygame.image.load("assets/field.png").convert_alpha()
        self.field_image = pygame.transform.scale(self.field_image, (WIDTH, HEIGHT))

        self.player1_image = pygame.image.load("assets/right_player_img.png").convert_alpha()
        self.player1_image = pygame.transform.scale(self.player1_image, (PLAYER_SIZE, PLAYER_SIZE))

        self.player2_image = pygame.image.load("assets/left_player_img.png").convert_alpha()
        self.player2_image = pygame.transform.scale(self.player2_image, (PLAYER_SIZE, PLAYER_SIZE))

        self.ball_image = pygame.image.load("assets/football.png").convert_alpha()
        self.ball_image = pygame.transform.scale(self.ball_image, (BALL_SIZE*2, BALL_SIZE*2))


    def __init__(self, render_mode=True):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT)) if render_mode else None
        self.clock = pygame.time.Clock()
        self.gk1 = pygame.Rect(10, HEIGHT // 2, PLAYER_SIZE, PLAYER_SIZE)  # Left goal
        self.gk2 = pygame.Rect(WIDTH - 30, HEIGHT // 2, PLAYER_SIZE, PLAYER_SIZE)  # Right goal
        if render_mode == True:
            self.load_assets()
        self.render_mode = render_mode
        self.reset()

    def reset(self):
        self.p1 = pygame.Rect(100, HEIGHT // 2, PLAYER_SIZE, PLAYER_SIZE)
        self.p2 = pygame.Rect(WIDTH - 100, HEIGHT // 2, PLAYER_SIZE, PLAYER_SIZE)
        self.ball = pygame.Rect(WIDTH // 2, HEIGHT // 2, BALL_SIZE, BALL_SIZE)
        self.ball_vel = [random.randint(-1,1),0]
        self.gk1 = pygame.Rect(10, HEIGHT // 2, PLAYER_SIZE, PLAYER_SIZE)  # Left goal
        self.gk2 = pygame.Rect(WIDTH - 30, HEIGHT // 2, PLAYER_SIZE, PLAYER_SIZE)  # Right goal
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



    stepcount = 0
    def step(self, action1, action2):
        old_dist = self._distance(self.p1.center, self.ball.center)
        scorer = 0
        #print(action1);
        self._move_player(self.p1, action1)
        self._move_player(self.p2, action2)
        self._handle_possession(action1=action1,action2=action2)

        reward = 0.0

        # Reward for gaining possession
        if self.possession == 1:
            reward += 0.01

        # Reward for moving closer to the ball
        new_dist = self._distance(self.p1.center, self.ball.center)
        if new_dist < old_dist:
            reward += 0.02
        
        if action1 == 4 and self.possession != 1:
            reward -= 0.5  # Penalty for kicking without possession


        # Optional: detect kicking (if action1 == 4 and possession)
        if action1 == 4 and self.possession == 1:
           # print("kicking logic reached")
            self.try_kick(self.p1, [1, 0])  # Kick right
            # Add extra reward if near the opponent's goal
            if self.ball.x > WIDTH * 0.7:
                reward += 0.23  # More incentive to kick near goal
            else:
                reward += 0.05  # Base reward for trying to kick

        if action2 == 4 and self.possession == 2:
           # print("kicking logic reached")
            self.try_kick(self.p2, [-1, 0])  # Kick right
            # Add extra reward if near the opponent's goal

        self._move_ball()

        goal_top = HEIGHT // 2 - GOAL_WIDTH // 2
        goal_bottom = HEIGHT // 2 + GOAL_WIDTH // 2

        # Check for own goal (ball crosses left boundary *and* is within goal height)
        if self.ball.left <= 0 and goal_top <= self.ball.y <= goal_bottom:
            print("Own goal")
            scorer = 2
            reward -= 0.5
            self.done = True

        # Check for scoring (ball crosses right boundary *and* is within goal height)
        elif self.ball.right >= WIDTH and goal_top <= self.ball.y <= goal_bottom:
            print("Scored a goal!")
            scorer = 1
            reward += 2.0
            if action1 == 4:  # last action was a kick
                reward += 0.5 
            self.done = True

        if self.stepcount % 5 == 0:
            #print("goalie move")
            self._move_goalkeepers()
        
        self.stepcount += 1

        if self.render_mode:
            self.render()

        return self.get_obs(), reward, self.done,scorer, {}
        
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

    def _handle_possession1(self):
        #if self.ball_vel == [0, 0]:  # Only allow possession if the ball is stationary
        if self.ball.colliderect(self.p1):
            self.possession = 1
            self.ball.center = self.p1.center
        elif self.ball.colliderect(self.p2):
            self.possession = 2
            self.ball.center = self.p2.center

    def _handle_possession(self,action1,action2):
   


        if self.ball.colliderect(self.gk1):
            self.ball_vel[0] = abs(self.ball_vel[0])  # bounce to the right

        if self.ball.colliderect(self.gk2):
            self.ball_vel[0] = -abs(self.ball_vel[0])

        
        # Check if current possessor is still close enough
        LOSS_PROBABILITY = 0.01  # 3% chance to lose possession randomly

        if self.possession == 1:
            if not self.ball.colliderect(self.p1) or random.random() < LOSS_PROBABILITY:
               # print("possession 1 lost")
                self.possession = 0
                return

        elif self.possession == 2:
            if not self.ball.colliderect(self.p2) or random.random() < LOSS_PROBABILITY:
               # print("possession 2 lost")
                self.possession = 0
                return

        # If already has possession and still close, don't change it
        if self.possession == 1:
            if action1 != 4:
                if action1 == 0: self.ball_vel[1] = -PLAYER_SPEED  # up
                elif action1 == 1: self.ball_vel[1] = PLAYER_SPEED  # down
                elif action1 == 2: self.ball_vel[0] = -PLAYER_SPEED  # left
                elif action1 == 3: self.ball_vel[0] = PLAYER_SPEED  # right
                #self.ball.center = self.p1.center
                self._move_ball()
            return
        
        if self.possession == 2:
            if action2 != 4:
                if action2 == 0: self.ball_vel[1] = -PLAYER_SPEED  # up
                elif action2 == 1: self.ball_vel[1] = PLAYER_SPEED  # down
                elif action2 == 2: self.ball_vel[0] = -PLAYER_SPEED  # left
                elif action2 == 3: self.ball_vel[0] = PLAYER_SPEED  # right
                #self.ball.center = self.p1.center
                self._move_ball()
            return

        # Determine if players are close enough to the ball
        p1_close = self.ball.colliderect(self.p1)
        p2_close = self.ball.colliderect(self.p2)

      #  print(p1_close,p2_close)

        # Both are in range
        if p1_close and p2_close:
          #  print("both close")
            self.possession = random.choice([1, 2])
            self.ball.center = self.p1.center if self.possession == 1 else self.p2.center

        elif p1_close:
            
            self.possession = 1
            self.ball.center = self.p1.center
           # print("p1 close",self.ball.center,self.p1.center)

        elif p2_close:
           # print("p2 close")
            self.possession = 2
            self.ball.center = self.p2.center

    def _move_goalkeepers(self):
        goal_top = HEIGHT // 2 - GOAL_WIDTH // 2
        goal_bottom = HEIGHT // 2 + GOAL_WIDTH // 2

        # Introduce randomness: slight delay or misjudgment
        reaction_chance = 0.5  # 90% chance to react correctly
        offset_range = 50  # pixels they might aim wrong

        # Goalkeeper 1
        if random.random() < reaction_chance:
            target_y1 = self.ball.centery + random.randint(-offset_range, offset_range)
            if target_y1 < self.gk1.centery:
                self.gk1.y -= PLAYER_SPEED
            elif target_y1 > self.gk1.centery:
                self.gk1.y += PLAYER_SPEED

        # Goalkeeper 2
        if random.random() < reaction_chance:
            target_y2 = self.ball.centery + random.randint(-offset_range, offset_range)
            if target_y2 < self.gk2.centery:
                self.gk2.y -= PLAYER_SPEED
            elif target_y2 > self.gk2.centery:
                self.gk2.y += PLAYER_SPEED

        # Clamp positions within the goal area
        self.gk1.y = max(goal_top, min(goal_bottom - PLAYER_SIZE, self.gk1.y))
        self.gk2.y = max(goal_top, min(goal_bottom - PLAYER_SIZE, self.gk2.y))

    def _move_ball(self):
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        if self.ball.top <= 0 or self.ball.bottom >= HEIGHT:
            self.ball_vel[1] *= -1
      

    def render(self):
        self.screen.blit(self.field_image, (0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), (0, HEIGHT//2 - GOAL_WIDTH//2, 10, GOAL_WIDTH))
        pygame.draw.rect(self.screen, (255, 255, 255), (WIDTH-10, HEIGHT//2 - GOAL_WIDTH//2, 10, GOAL_WIDTH))
        self.screen.blit(self.player1_image, self.p1.topleft)
        self.screen.blit(self.player2_image, self.p2.topleft)
        self.screen.blit(self.ball_image, self.ball.topleft)
        pygame.draw.rect(self.screen, (0, 15, 155), self.gk1)  # Blue for gk1
        pygame.draw.rect(self.screen, (155, 15, 0), self.gk2)  # Red for gk2
        pygame.display.flip()
        self.clock.tick(60)

