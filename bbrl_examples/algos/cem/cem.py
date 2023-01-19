import numpy as np
import gym


import torch
import torch.nn as nn
import hydra

from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl_examples.models.loggers import Logger
from bbrl_examples.models.actors import DiscreteDeterministicActor
from bbrl_examples.models.envs import create_no_reset_env_agent


class CovMatrix:
    def __init__(self, centroid: torch.Tensor, sigma, noise_multiplier):
        policy_dim = centroid.size()[0]
        self.noise = torch.diag(torch.ones(policy_dim) * sigma)
        self.cov = torch.diag(torch.ones(policy_dim) * torch.var(centroid)) + self.noise
        self.noise_multiplier = noise_multiplier

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def generate_weights(self, centroid, pop_size):
        dist = torch.distributions.MultivariateNormal(
            centroid, covariance_matrix=self.cov
        )
        weights = [dist.sample() for _ in range(pop_size)]
        return weights

    def update_covariance(self, elite_weights) -> None:
        self.cov = torch.cov(elite_weights.T) + self.noise


# Create the PPO Agent
def create_CEM_agent(cfg, env_agent):
    obs_size, act_size = env_agent.get_obs_and_actions_sizes()
    action_agent = DiscreteDeterministicActor(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    ev_agent = Agents(env_agent, action_agent)
    eval_agent = TemporalAgent(ev_agent)
    eval_agent.seed(cfg.algorithm.seed)

    return eval_agent


def make_gym_env(env_name):
    env = gym.make(env_name)
    return env


def run_cem(cfg):
    # 1)  Build the  logger
    torch.manual_seed(cfg.algorithm.seed)
    # logger = Logger(cfg)
    eval_env_agent = create_no_reset_env_agent(cfg)

    pop_size = cfg.algorithm.pop_size

    assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0
    eval_agent = create_CEM_agent(cfg, eval_env_agent)

    centroid = torch.nn.utils.parameters_to_vector(eval_agent.parameters())
    matrix = CovMatrix(
        centroid,
        cfg.algorithm.sigma,
        cfg.algorithm.noise_multiplier,
    )

    best_score = -np.inf

    # 7) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        matrix.update_noise()
        scores = []
        weights = matrix.generate_weights(centroid, pop_size)

        for i in range(pop_size):
            workspace = Workspace()
            w = weights[i]
            torch.nn.utils.vector_to_parameters(w, eval_agent.parameters())

            eval_agent(workspace, t=0, stop_variable="env/done")
            episode_lengths = workspace["env/done"].float().argmax(0) + 1
            arange = torch.arange(cfg.algorithm.n_envs)
            mean_reward = (
                workspace["env/cumulated_reward"][episode_lengths - 1, arange]
                .mean()
                .item()
            )
            # ---------------------------------------------------
            scores.append(mean_reward)

            if mean_reward >= best_score:
                best_score = mean_reward
                # best_params = weights[i]

            if cfg.verbose > 0:
                print(f"Indiv: {i + 1} score {scores[i]:.2f}")

        print("Best score: ", best_score)
        # Keep only best individuals to compute the new centroid
        elites_idxs = np.argsort(scores)[-cfg.algorithm.elites_nb :]
        elites_weights = [weights[k] for k in elites_idxs]
        elites_weights = torch.cat(
            [torch.tensor(w).unsqueeze(0) for w in elites_weights], dim=0
        )
        centroid = elites_weights.mean(0)

        # Update covariance
        matrix.update_noise()
        matrix.update_covariance(elites_weights)


@hydra.main(
    config_path="./configs/",
    config_name="cem_cartpole.yaml",
    version_base="1.1",
)
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_cem(cfg)


if __name__ == "__main__":
    main()
