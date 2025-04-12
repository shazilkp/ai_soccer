import pickle
import neat
import pygame
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import re
from datetime import datetime
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


def plot_heatmap(positions, gen_num):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=(32, 24), range=[[0, WIDTH], [0, HEIGHT]])

    plt.clf()
    plt.imshow(heatmap.T, origin='lower', cmap='magma', interpolation='nearest', extent=[0, WIDTH, 0, HEIGHT])
    plt.colorbar(label='Visit Frequency')
    plt.title(f"Player 1 Heatmap - Generation {gen_num}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"heatmaps"

    os.makedirs(dirname, exist_ok=True)
    filename = f"{dirname}/heatmap_gen{gen_num}.png"
    plt.savefig(filename)
    print(f"Heatmap saved: {filename}")



# Visualize a genome for 5 episodes
def visualize_agent(net, generation_number):
    p1_positions = []  # Track across all 5 episodes
    obs_log = []
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
            #action2 = random_opponent_action()
            print(action1,action2,env.possession)
            obs, reward, done, scorer, _ = env.step(action1, action2)

            p1_positions.append(env.p1.center)
            if generation_number == "winner":
                obs_log.append(obs)
            # Draw overlay info
            draw_text(env.screen, f"Gen {generation_number}, Episode {episode}", 10, 10)

            pygame.display.update()

        print(f"Gen {generation_number} - Episode {episode} ended. Final reward: {reward}")
        plot_heatmap(p1_positions, generation_number)
        if generation_number == "winner":
                with open('obs_log_winner.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["p1_x", "p1_y", "p2_x", "p2_y", "ball_x", "ball_y", "ball_vel_0", "ball_vel_1", "possession"])
                    writer.writerows(obs_log)
                print("Logged observations to obs_log_winner.csv")
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
