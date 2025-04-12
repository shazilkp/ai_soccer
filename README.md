# ⚽ AI Football with NEAT

A Python project that uses the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm to evolve an intelligent football-playing agent in a dynamic, adversarial 2D environment.

## 📌 Overview

This project demonstrates how an agent can learn to make strategic decisions—such as positioning, chasing, intercepting, and kicking—using **evolutionary reinforcement learning**. The AI learns by playing against a **rule-based opponent** in a custom Pygame environment.

It was developed as part of an **Intro to AI course project**, focused on evolving intelligent behaviors through neuroevolution instead of classical supervised learning or hardcoded logic.

## 🧠 Technologies Used

- `Python 3.x`
- [`neat-python`](https://github.com/CodeReclaimers/neat-python) – for evolving neural networks
- `Pygame` – for 2D game simulation
- `Matplotlib` – for plotting fitness and performance stats
- `CSV` – for logging observations and results

## 🎮 Environment

The simulation includes:
- Two players: **Player 1 (AI-controlled)** and **Player 2 (Rule-based)** and 2 rule based goalkeepers
- A ball that can be passed, chased, or kicked
- A reward-based system to train decision-making

## 🔍 Inputs to NEAT Agent

The neural network is trained on the following observations:
- Normalized (x, y) positions of both players and the ball
- Ball velocity (x, y)
- Ball possession flag: `0 = None`, `1 = Player 1`, `2 = Player 2`

## 🧪 Fitness Evaluation

Fitness is based on:
- Goals scored
- Possession time
- Defensive actions
- Reward shaping to encourage intelligent play

Each generation logs:
- Max fitness
- Goals scored vs. conceded
- Goal scorer identification (binary/int value)
- Optional: Observation logs for analysis

## 📊 Results

Trained agents show clear behavioral improvements:
- Better positioning and pursuit
- Goal-scoring strategies
- Defensive interception
- Increased match fitness across generations

Performance is visualized via:
- Fitness graphs
- Goal statistics (scored vs. conceded)
- Heatmaps and positional plots

## 📁 Project Structure

```
├── env.py                  # Environment simulation using Pygame
├── main.py                 # Core NEAT training loop
├── config-feedforward.txt  # NEAT configuration
├── obs_log_gen1.csv        # Sample logged observations
├── fitness_log_*.csv       # Fitness scores over generations
├── best_gen_gen*.pkl       # Saved top genomes
├── winner.pkl              # Final best performing genome
└── README.md               # You're here!
```

## 🚀 Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/shazilkp/ai_soccer.git
   cd ai_soccer
   ```

2. **Install dependencies:**

   ```bash
   pip install neat-python pygame matplotlib
   ```

3. **Run the training:**

   ```bash
   python train_neat.py
   python agent_test.py
   ```
