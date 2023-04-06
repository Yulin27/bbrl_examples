import torch
import torch.nn as nn

from bbrl_examples.models.actors import BaseActor

from torch.distributions.normal import Normal
from torch.distributions import Bernoulli, Independent
from bbrl.utils.distributions import SquashedDiagGaussianDistribution

from bbrl_examples.models.shared_models import (
    build_mlp,
    build_backbone,
    build_ortho_mlp,
    build_ortho_backbone,
)

from bbrl.agents.agent import Agent


class ActorAgent(Agent):
    """Choose an action (either according to p(a_t|s_t) when stochastic is true,
    or with argmax if false.
    """

    def __init__(self):
        super().__init__()

    def forward(self, t, stochastic, **kwargs):
        probs = self.get(("action_probs", t))
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)


class BernoulliActor(Agent):
    def __init__(self, state_dim, hidden_layers):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [1]
        self.model = build_mlp(
            layers, activation=nn.ReLU(), output_activation=nn.Sigmoid()
        )

    def forward(self, t, stochastic=False, **kwargs):
        obs = self.get(("env/env_obs", t))
        mean = self.model(obs)
        dist = Bernoulli(mean)
        self.set(("entropy", t), dist.entropy())
        if stochastic:
            action = dist.sample().int().squeeze(-1)
        else:
            act = mean.lt(0.5)
            action = act.squeeze(-1)
        # print(f"stoch:{stochastic} obs:{obs} mean:{mean} dist:{dist} action:{action}")
        log_prob = dist.log_prob(action.float()).sum(axis=-1)
        self.set(("action", t), action)
        self.set(("action_logprobs", t), log_prob)

    def predict_action(self, obs, stochastic=False):
        mean = self.model(obs)
        dist = Bernoulli(mean)
        if stochastic:
            act = dist.sample().int()
            return act
        else:
            act = mean.lt(0.5)
        return act


class ProbAgent(Agent):
    def __init__(self, state_dim, hidden_layers, n_action):
        super().__init__(name="prob_agent")
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [n_action], activation=nn.Tanh()
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        action_probs = torch.softmax(scores, dim=-1)
        assert not torch.any(torch.isnan(action_probs)), "Nan Here"
        self.set(("action_probs", t), action_probs)
        entropy = torch.distributions.Categorical(action_probs).entropy()
        self.set(("entropy", t), entropy)


class ActionAgent(Agent):
    def __init__(self):
        super().__init__()

    def forward(self, t, stochastic, **kwargs):
        probs = self.get(("action_probs", t))
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)


class DiscreteActor(BaseActor):
    def __init__(self, state_dim, hidden_size, n_actions):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_size) + [n_actions], activation=nn.ReLU()
        )

    def get_distribution(self, obs):
        scores = self.model(obs)
        probs = torch.softmax(scores, dim=-1)
        return torch.distributions.Categorical(probs), scores

    def forward(
        self, t, stochastic=False, predict_proba=False, compute_entropy=False, **kwargs
    ):
        """
        Compute the action given either a time step (looking into the workspace)
        or an observation (in kwargs)
        If predict_proba is true, the agent takes the action already written in the workspace and adds its probability
        Otherwise, it writes the new action
        """
        if "observation" in kwargs:
            observation = kwargs["observation"]
        else:
            observation = self.get(("env/env_obs", t))
        dist, scores = self.get_distribution(observation)
        probs = torch.softmax(scores, dim=-1)

        if compute_entropy:
            entropy = dist.entropy()
            self.set(("entropy", t), entropy)

        if predict_proba:
            action = self.get(("action", t))
            log_prob = probs[torch.arange(probs.size()[0]), action].log()
            self.set(("logprob_predict", t), log_prob)
        else:
            if stochastic:
                action = dist.sample()
            else:
                action = scores.argmax(1)

            log_probs = probs[torch.arange(probs.size()[0]), action].log()

            self.set(("action", t), action)
            self.set(("action_logprobs", t), log_probs)

    def predict_action(self, obs, stochastic=False):
        dist, scores = self.get_distribution(obs)

        if stochastic:
            action = dist.sample()
        else:
            action = scores.argmax(0)
        return action


# All the actors below use a Gaussian policy, that is the output is Normal distribution


class StochasticActor(BaseActor):
    def __init__(self):
        super().__init__()

    def get_distribution(self, obs: torch.Tensor):
        raise NotImplementedError

    def forward(
        self, t, stochastic=False, predict_proba=False, compute_entropy=False, **kwargs
    ):
        """
        Compute the action given either a time step (looking into the workspace)
        or an observation (in kwargs)
        If predict_proba is true, the agent takes the action already written in the workspace and adds its probability
        Otherwise, it writes the new action
        """
        obs = self.get(("env/env_obs", t))
        dist, mean = self.get_distribution(obs)

        if compute_entropy:
            self.set(("entropy", t), dist.entropy())

        if predict_proba:
            action = self.get(("action", t))
            self.set(("logprob_predict", t), dist.log_prob(action))
        else:
            action = dist.sample() if stochastic else mean

            self.set(("action", t), action)
            self.set(("action_logprobs", t), dist.log_prob(action))

    def predict_action(self, obs, stochastic=False):
        """Predict just one action (without using the workspace)"""
        dist, mean = self.get_distribution(obs)
        return dist.sample() if stochastic else mean


class TunableVarianceContinuousActor(StochasticActor):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(layers, activation=nn.ReLU())
        init_variance = torch.randn(action_dim, 1).transpose(0, 1)
        self.std_param = nn.parameter.Parameter(init_variance)
        self.soft_plus = torch.nn.Softplus()

    def get_distribution(self, obs: torch.Tensor):
        mean = self.model(obs)
        # std must be positive
        return Independent(Normal(mean, self.soft_plus(self.std_param[:, 0])), 1), mean


class TunableVarianceContinuousActorExp(StochasticActor):
    """
    A variant of the TunableVarianceContinuousActor class where, instead of using a softplus on the std,
    we exponentiate it
    """

    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(layers, activation=nn.Tanh())
        self.std_param = nn.parameter.Parameter(torch.randn(1, action_dim))

    def get_distribution(self, obs: torch.Tensor):
        mean = self.model(obs)
        std = torch.clamp(self.std_param, -20, 2)
        return Independent(Normal(mean, torch.exp(std)), 1), mean


class StateDependentVarianceContinuousActor(StochasticActor):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        backbone_dim = [state_dim] + list(hidden_layers)
        self.layers = build_backbone(backbone_dim, activation=nn.Tanh())
        self.backbone = nn.Sequential(*self.layers)

        self.last_mean_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.last_std_layer = nn.Linear(hidden_layers[-1], action_dim)

    def get_distribution(self, obs: torch.Tensor):
        backbone_output = self.backbone(obs)
        mean = self.last_mean_layer(backbone_output)
        std_out = self.last_std_layer(backbone_output)
        std = torch.exp(std_out)
        return Independent(Normal(mean, std), 1), mean


class ConstantVarianceContinuousActor(StochasticActor):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(layers, activation=nn.Tanh())
        self.std_param = 2

    def get_distribution(self, obs: torch.Tensor):
        mean = self.model(obs)
        return Normal(mean, self.std_param), mean


class SquashedGaussianActor(StochasticActor):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        backbone_dim = [state_dim] + list(hidden_layers)
        self.layers = build_backbone(backbone_dim, activation=nn.Tanh())
        self.backbone = nn.Sequential(*self.layers)
        self.last_mean_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.last_std_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)

    def get_distribution(self, obs: torch.Tensor):
        backbone_output = self.backbone(obs)
        mean = self.last_mean_layer(backbone_output)
        std_out = self.last_std_layer(backbone_output)

        std_out = std_out.clamp(-20, 2)  # as in the official code
        std = torch.exp(std_out)
        return self.action_dist.make_distribution(mean, std), mean

    def test(self, obs, action):
        action_dist = self.get_distribution(obs)
        return action_dist.log_prob(action)


class TunableVariancePPOActor(StochasticActor):
    """
    The official PPO actor uses Tanh activation functions and orthogonal initialization
    """

    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_ortho_mlp(layers, activation=nn.Tanh())
        init_variance = torch.randn(1, action_dim)
        self.std_param = nn.parameter.Parameter(init_variance)
        self.soft_plus = torch.nn.Softplus()

    def get_distribution(self, obs: torch.Tensor):
        mean = self.model(obs)
        # std must be positive
        return Independent(Normal(mean, self.soft_plus(self.std_param[:, 0])), 1), mean
    
class TunableVariancePPOLSTMActor(StochasticActor):
    """
    The official PPO actor uses Tanh activation functions and orthogonal initialization
    """
    def __init__(self, state_dim, hidden_dim, nb_layers, action_dim):
        super().__init__()
        self.is_q_function = False
        self.hidden_dim = hidden_dim
        self.model = nn.LSTM(state_dim, hidden_dim, nb_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.nb_layers = nb_layers
        self.batch_size = 1
        self.h = torch.zeros(nb_layers, self.batch_size, hidden_dim)
        self.c = torch.zeros(nb_layers, self.batch_size, hidden_dim)
        self.init_weights()

    def init_weights(self):
        # Orthogonal initialization for LSTM weights
        for name, param in self.model.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

    def forward(self, t, choose_action=True, **kwargs):
        obs = self.get(("env/env_obs", t))
        self.batch_size = obs.shape[0]
        lstm_out, (h, c) = self.model(obs.unsqueeze(0), (self.h, self.c))

        # Extract the v-value from LSTM output
        v_value = lstm_out[:, -1, :]  # Extract the last timestep's output as the v-value
        q_values = self.fc(v_value)
        # Apply Tanh activation to q_values
        q_values = torch.tanh(q_values)
        self.set(("v_value", t), v_value)
        self.set(("q_values", t), q_values)
        if choose_action:
            action = q_values.argmax(1)
            self.set(("action", t), action)
            
    def refresh_internal_values(self):
        self.h = torch.zeros(self.nb_layers, self.batch_size, self.hidden_dim)
        self.c = torch.zeros(self.nb_layers, self.batch_size, self.hidden_dim)

    def get_distribution(self, obs: torch.Tensor):
        mean = self.forward(obs)
        # std must be positive
        return Independent(Normal(mean, self.soft_plus(self.std_param)), 1), mean
