import pickle
import neat
import pygame
import numpy as np
import os
import re
from env import SoccerEnv  # Your custom env

WIDTH, HEIGHT = 640, 480
PLAYER_SIZE = 20
BALL_SIZE = 10
PLAYER_SPEED = 5
BALL_SPEED = 6
GOAL_WIDTH = 80

# Load the NEAT config
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config-feedforward.txt"
)

# Random policy for player 2
def random_opponent_action():
    return np.random.randint(0, 5)

def get_agent2_action(env):
    p2 = env.p2
    p1 = env.p1
    ball = env.ball
    possession = env.possession

    def direction_to(src, dst):
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2  # right or left
        else:
            return 1 if dy > 0 else 0  # down or up

    # If agent 2 has possession, try to kick if facing goal
    if possession == 2:
        goal_y = HEIGHT // 2
        print("agent2 ball dist",abs(p2.centery - goal_y))
        if abs(p2.centery - goal_y) < 40:
            print("in agent 2 possesion")
            return 4  # kick
        else:
            return direction_to(p2.center, (0, goal_y))

    # If near the ball, try to gain possession
    if env._distance(p2.center, ball.center) < 0.05:
        return direction_to(p2.center, ball.center)

    # If opponent has possession, intercept them
    if possession == 1:
        return direction_to(p2.center, p1.center)

    # Default: move towards ball
    return direction_to(p2.center, ball.center)


def rule_based_agent2(env):
    p2 = env.p2
    ball = env.ball
    goal_y_center = HEIGHT // 2

    # If player 2 is in possession and aligned with the goal, kick
    if env.possession == 2:
        if abs(p2.centery - goal_y_center) < GOAL_WIDTH // 2:
            return 4  # kick (handled by env if possession)
    
    # Otherwise move towards the ball
    if abs(ball.x - p2.x) > abs(ball.y - p2.y):
        return 3 if ball.x > p2.x else 2  # move right or left
    else:
        return 1 if ball.y > p2.y else 0  # move down or up

# Render overlay text (generation, episode)
def draw_text(screen, text, x, y, size=24, color=(255, 255, 255)):
    font = pygame.font.SysFont("Arial", size)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

# Visualize a genome for 5 episodes
def visualize_agent(net, generation_number):
    for episode in range(1, 6):
        env = SoccerEnv(render_mode=True)
        obs = env.reset()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            output = net.activate(obs)
            action1 = np.argmax(output)
            action2 = get_agent2_action(env)
            print(action1,action2)
            obs, reward, done, _ = env.step(action1, action2)

            # Draw overlay info
            draw_text(env.screen, f"Gen {generation_number}, Episode {episode}", 10, 10)

            pygame.display.update()

        print(f"Gen {generation_number} - Episode {episode} ended. Final reward: {reward}")
        pygame.time.wait(1000)  # Pause briefly between episodes
        pygame.display.quit()

# Load all best_gen_genXX.pkl files and run them
def run_all_best_genomes():
    files = [f for f in os.listdir() if re.match(r"best_gen_gen\d+\.pkl", f)]
    files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))  # Sort by generation number

    for file in files:
        gen_num = int(re.findall(r'\d+', file)[0])
        with open(file, "rb") as f:
            genome = pickle.load(f)

        print(f"\n🔍 Running visualization for Generation {gen_num}")
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        visualize_agent(net, gen_num)
    
    if os.path.exists("winner.pkl"):
        with open("winner.pkl", "rb") as f:
            winner_genome = pickle.load(f)

        print(f"\n🏆 Running visualization for Final Winner")
        net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
        visualize_agent(net, "winner")

run_all_best_genomes()
