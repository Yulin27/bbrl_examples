from bbrl import get_arguments, get_class, instantiate_class
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl_examples.wrappers.env_wrappers import FilterWrapper, DelayWrapper

class AutoResetEnvAgent(AutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs)
        env = instantiate_class(cfg.gym_env)
        env.seed(cfg.algorithm.seed)
        # Appliquer le wrapper FilterWrapper pour enlever les informations de vitesse
        env = FilterWrapper(env)

        # Appliquer le wrapper DelayWrapper pour renvoyer les informations au bout de N pas de temps
        env = DelayWrapper(env)
        del env


class NoAutoResetEnvAgent(NoAutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments without auto-reset
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs)
        env = instantiate_class(cfg.gym_env)
        env.seed(cfg.algorithm.seed)
        # Appliquer le wrapper FilterWrapper pour enlever les informations de vitesse
        env = FilterWrapper(env)

        # Appliquer le wrapper DelayWrapper pour renvoyer les informations au bout de N pas de temps
        env = DelayWrapper(env, 200)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env


def create_env_agents(cfg):
    """Create the environment agents"""
    train_env_agent = AutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )

    eval_env_agent = NoAutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.nb_evals,
        cfg.algorithm.seed,
    )
    return train_env_agent, eval_env_agent


def create_no_reset_env_agent(cfg):
    eval_env_agent = NoAutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.nb_evals,
        cfg.algorithm.seed,
    )
    return eval_env_agent
