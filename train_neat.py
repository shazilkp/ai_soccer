import neat
import numpy as np
from env import SoccerEnv
import os
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

generation_counter = [0]  # use list so it can be mutated inside the function

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
           # a2 = np.random.choice([0, 1, 2, 3, 4])  # Random policy for player 2
            a2 = rule_based_agent2(env)
            print(a2)
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

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run_neat(config_path)
