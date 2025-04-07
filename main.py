
from env import SoccerEnv
import pygame
import time
import random

env = SoccerEnv(render_mode=True)
EPISODES = 5

for ep in range(EPISODES):
    print(f"Episode {ep + 1}")
    obs = env.reset()
    done = False
    kick1 = False
    kick2 = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    kick1 = True
                if event.key == pygame.K_RETURN:
                    kick2 = True

        player1_pos = obs[0:2]
        player2_pos = obs[2:4]
        ball_pos = obs[4:6]

        #a1 = random.choice([0, 1, 2, 3])
        a1 = 3
        a2 = random.choice([0, 1, 2, 3])

        obs, reward, done, _ = env.step(a1, a2)
        print(f"P1 Pos: ({obs[0]:.3f}, {obs[1]:.3f})")
        print(f"P2 Pos: ({obs[2]:.3f}, {obs[3]:.3f})")
        print(f"Ball Pos: ({obs[4]:.3f}, {obs[5]:.3f}) | Ball Vel: ({obs[6]:.3f}, {obs[7]:.3f})")
        print(f"Possession: {int(obs[8] * 2)}")  # 0=none, 1=p1, 2=p2
        print("-" * 40)


        if kick1:
            env.try_kick(env.p1, [1, 0])  # right
            kick1 = False
        if kick2:
            env.try_kick(env.p2, [-1, 0])  # left
            kick2 = False

        
        time.sleep(0.05)

pygame.quit()