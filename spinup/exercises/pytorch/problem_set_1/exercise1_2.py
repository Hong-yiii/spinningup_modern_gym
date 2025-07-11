import torch
import torch.nn as nn
import numpy as np
from spinup.exercises.pytorch.problem_set_1 import exercise1_1
from spinup.exercises.pytorch.problem_set_1 import exercise1_2_auxiliary

"""

Exercise 1.2: PPO Gaussian Policy

You will implement an MLP diagonal Gaussian policy for PPO by
writing an MLP-builder, and a few other key functions.

Log-likelihoods will be computed using your answer to Exercise 1.1,
so make sure to complete that exercise before beginning this one.

"""

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Build a multi-layer perceptron in PyTorch.

    Args:
        sizes: Tuple, list, or other iterable giving the number of units
            for each layer of the MLP. 

        activation: Activation function for all layers except last.

        output_activation: Activation function for last layer.

    Returns:
        A PyTorch module that can be called to give the output of the MLP.
        (Use an nn.Sequential module.)

    """

    layers = []

    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            layers.append(activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)

class DiagonalGaussianDistribution:

    def __init__(self, mu, log_std):
        self.mu = mu
        self.log_std = log_std

    def sample(self):
        """
        Returns:
            A PyTorch Tensor of samples from the diagonal Gaussian distribution with
            mean and log_std given by self.mu and self.log_std.
        """
        std = torch.exp(self.log_std)
        noise = torch.randn_like(self.mu)
        return self.mu + std * noise

    def log_prob(self, value):
        # Add batch dimension if not present
        if value.dim() == 1:
            value = value.unsqueeze(0)
        if self.mu.dim() == 1:
            mu = self.mu.unsqueeze(0)
            log_std = self.log_std.unsqueeze(0)
        else:
            mu = self.mu
            log_std = self.log_std
        return exercise1_1.gaussian_likelihood(value, mu, log_std)

    def entropy(self):
        return 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std.sum(axis=-1)
    #=========================================================================================#


class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        """
        Initialize an MLP Gaussian Actor by making a PyTorch module for computing the
        mean of the distribution given a batch of observations, and a log_std parameter.

        Make log_std a PyTorch Parameter with the same shape as the action vector, 
        independent of observations, initialized to [-0.5, -0.5, ..., -0.5].
        (Make sure it's trainable!)
        """
        # Initialize log_std as a trainable parameter, nn.parameter is responsible for making it trainable
        self.log_std = nn.Parameter(torch.ones(act_dim, dtype=torch.float32) * -0.5)

        self.mu_net = mlp(
            sizes=[obs_dim] + list(hidden_sizes) + [act_dim],
            activation=activation,
            output_activation=nn.Identity # No activation for the last layer, output is not squished
        )

    #================================(Given, ignore)==========================================#
    def forward(self, obs, act=None):
        mu = self.mu_net(obs)
        pi = DiagonalGaussianDistribution(mu, self.log_std)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a
    #=========================================================================================#



if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """

    from spinup.algos.pytorch.ppo.ppo import ppo
    from spinup.exercises.common import print_result
    from functools import partial
    import gymnasium as gym  # Updated to gymnasium
    import os
    import pandas as pd
    import psutil
    import time

    # Create experiment directory if it doesn't exist
    logdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                         "training_results", 
                         str(int(time.time())))
    os.makedirs(logdir, exist_ok=True)

    # Create a class that matches the expected type
    class ActorCritic(exercise1_2_auxiliary.ExerciseActorCritic):
        def __init__(self, observation_space, action_space, hidden_sizes=(64,64), activation=nn.Tanh):
            super().__init__(observation_space, action_space, hidden_sizes, activation, actor=MLPGaussianActor)
    
    ppo(env_fn = lambda : gym.make('InvertedPendulum-v4'),
        actor_critic=ActorCritic,
        ac_kwargs=dict(hidden_sizes=(64,)),
        steps_per_epoch=4000, epochs=20, logger_kwargs=dict(output_dir=logdir))

    # Get scores from last five epochs to evaluate success.
    data = pd.read_table(os.path.join(logdir,'progress.txt'))
    last_scores = data['AverageEpRet'][-5:]

    # Your implementation is probably correct if the agent has a score >500,
    # or if it reaches the top possible score of 1000, in the last five epochs.
    correct = bool(np.mean(last_scores) > 500 or np.max(last_scores)==1e3)  # Convert numpy bool to Python bool
    print_result(correct)