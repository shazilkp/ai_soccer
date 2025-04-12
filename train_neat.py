import neat
import numpy as np
from env import SoccerEnv
import os
import matplotlib.pyplot as plt
import csv
from datetime import datetime
#import visualize

EPISODES = 5
WIDTH, HEIGHT = 640, 480
PLAYER_SIZE = 20
BALL_SIZE = 10
PLAYER_SPEED = 5
BALL_SPEED = 6
GOAL_WIDTH = 80

# Convert output to discrete action (0-4)
def interpret_output(output):
    return np.argmax(output)

max_fitnesses = []
generation_counter = [0]  # use list so it can be mutated inside the function

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
        #print("agent2 ball dist",abs(p2.centery - goal_y))
        if abs(p2.centery - goal_y) < 40:
            #print("in agent 2 possesion")
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


def eval_genomes(genomes, config):
    generation_counter[0] += 1
    print(f"\n=== Generation {generation_counter[0]} ===")

    best_genome = None
    best_fitness = float('-inf')

    for genome_id, genome in genomes:
        eval_genome(genome, config)
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome = genome

    max_fitnesses.append(best_fitness)

    # Save best genome every 10 generations
    if generation_counter[0] % 10 == 0:
        import pickle
        with open(f"best_gen_gen{generation_counter[0]}.pkl", "wb") as f:
            pickle.dump(best_genome, f)
        print(f"✔ Saved best genome of generation {generation_counter[0]} (fitness={best_fitness})")


def eval_genome(genome, config):
    print("Evaluating a genome...") 
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = SoccerEnv(render_mode=False)
    total_reward = 0.0
    MAX_STEPS = 300

    for episode in range(EPISODES):
        obs = env.reset()
        done = False
        step = 0

        while not done and step < MAX_STEPS:
            inputs = obs
            output = net.activate(inputs)
            a1 = interpret_output(output)
            a2 = np.random.choice([0, 1, 2, 3, 4])  # Random policy for player 2
            #a2 = get_agent2_action(env)
            #print(a2)
            try:
                #if a1 == 4 and env.possession == 1:
                #    print("Agent tried to kick!")
                 #   env.try_kick(env.p1, [1, 0])  # Right direction

                obs, reward, done, _ = env.step(a1, a2)
                total_reward += reward
            except Exception as e:
                print(f"Error during step: {e}")
                break

            step += 1

        if step >= MAX_STEPS:
            print(f"Episode {episode+1} ended due to timeout.")
            total_reward -= 0.2  # Penalty for not scoring
        else:
            print(f"Episode {episode+1} ended with goal.")

    # Avoid 0 fitness to prevent stagnation
    genome.fitness = total_reward / EPISODES + 1e-6



def run_neat(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)

    print("\nBest genome:\n{}".format(winner))

    # Save winner
    with open("winner.pkl", "wb") as f:
        import pickle
        pickle.dump(winner, f)

    # Optionally visualize
    # visualize.draw_net(config, winner, True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"fitness_log_{timestamp}.csv"
    png_filename = f"fitness_plot_{timestamp}.png"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Generation", "MaxFitness"])
        for i, fitness in enumerate(max_fitnesses):
            writer.writerow([i + 1, fitness])
    print(f"✔ Fitness data saved to {csv_filename}")

    plt.figure(figsize=(10, 5))
    print(max_fitnesses)
    plt.plot(max_fitnesses, label='Max Fitness per Generation')
    plt.axhline(y=config.fitness_threshold, color='r', linestyle='--', label='Fitness Threshold')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Progression Over Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_filename)
    plt.show()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run_neat(config_path)
