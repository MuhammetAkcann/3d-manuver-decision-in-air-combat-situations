"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gym
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from PPO.network import FeedForwardNN

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class PPO:
    """
        This is the PPO class we will use as our model in main.py
    """

    def __init__(self, policy_class, env, rival_actor, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.
            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
            Returns:
                None
        """
        # Make sure the environment is compatible with our code
        assert (type(env.observation_space) == gym.spaces.Box)
        assert (type(env.action_space) == gym.spaces.Box)

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.rival_policy = FeedForwardNN(self.obs_dim, self.act_dim)
        self.rival_policy.load_state_dict(torch.load(rival_actor))

        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim)  # ALG STEP 1
        self.critic = policy_class(self.obs_dim, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
            'avg_batch_losses': [],
            'avg_batch_rewards': [],
            'avg_batch_rewards_rival': [],
            'batch_rews_rival': [],
            'batch_win': [],
            'batch_lose': [],
            'batch_draw': []
        }

    def arrange_observation(obs, x):
        """
            x parametresine göre yaw pitch roll position çikarilacak
            y parametresine göre noise eklenip çikarilacak
            eğer x e göre shadowlama yapildiysa rivalin eski posisyonu eklencek
        """
        pass

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.
            Parameters:
                total_timesteps - the total number of timesteps to train for
            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while t_so_far < total_timesteps:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_win, batch_lose, batch_draw = self.rollout()  # ALG STEP 3
            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()  # ALG STEP 5

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), 'PPO/weights/backup/ppo_actor.pth')
                torch.save(self.critic.state_dict(), 'PPO/weights/backup/ppo_critic.pth')

    def rollout(self):
        """
            Too many transformers references, I'm sorry. This is where we collect the batch of data
            from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
            of data each time we iterate the actor/critic networks.
            Parameters:
                None
            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rews_rival = []
        batch_lens = []
        batch_win = []
        batch_lose = []
        batch_draw = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode

        t = 0  # Keeps track of how many timesteps we've run so far this batch
        episode_number = 0
        # Keep simulating until we've run more than or equal to specified timesteps per batch

        numb_win = 0
        numb_lose = 0
        numb_draw = 0
        while t < self.timesteps_per_batch:
            # print("t:", t)
            episode_number += 1
            ep_rews = []  # rewards collected per episode
            ep_rews_rival = []
            # Reset the environment. sNote that obs is short for observation.
            obs, rival_obs = self.env.reset()

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()
                    pass

                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.

                rival_action = self.rival_policy(rival_obs).detach().numpy()

                action, log_prob = self.get_action(obs)

                both_action = np.concatenate((action, rival_action), axis=None)

                obs, rival_obs, rew, done, rival_rew = self.env.step(both_action)
                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                ep_rews_rival.append(rival_rew)

                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done != 0:
                    break

            self.env.close()

            if done == 1:
                numb_win += 1
            elif done == -1:
                numb_lose += 1
            else:
                numb_draw += 1
            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_rews_rival.append(ep_rews_rival)

            t += 1  # Increment timesteps ran this batch so far

        batch_win.append(numb_win)
        batch_lose.append(numb_lose)
        batch_draw.append(numb_draw)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4
        # batch_rtgs_rival = self.compute_rtgs(batch_rews_rival)
        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_rews_rival'] = batch_rews_rival
        self.logger['batch_lens'] = batch_lens

        self.logger['batch_win'] = batch_win
        self.logger['batch_lose'] = batch_lose
        self.logger['batch_draw'] = batch_draw

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_win, batch_lose, batch_draw #, batch_rtgs_rival

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.
            Parameters:
                obs - the observation at the current timestep
            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        mean = self.actor(obs)
        # print("mean:", mean)
        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)
        # print("dist:", dist)

        # Sample an action from the distribution
        action = dist.sample()
        # print("action:", action)

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)
        # print("log_prob:", log_prob)
        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.
            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters
            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.
            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 8  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600  # Max number of timesteps per episode
        self.n_updates_per_iteration = 5  # Number of times to update actor/critic per iteration
        self.lr = 0.005  # Learning rate of actor optimizer
        self.gamma = 0.95  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = None  # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed is not None:
            # Check if our seed is valid first
            assert (type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.
            Parameters:
                None
            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.mean(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_ep_rews_last_50 = [np.mean(ep_rews[-50:]) for ep_rews in self.logger['batch_rews']]
        avg_ep_rews_rival = np.mean([np.mean(ep_rews) for ep_rews in self.logger['batch_rews_rival']])
        avg_ep_rews_rival_last_50 = [np.mean(ep_rews[-50:]) for ep_rews in self.logger['batch_rews_rival']]
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = round(avg_ep_lens, 2)
        avg_ep_rews = round(avg_ep_rews, 2)
        avg_ep_rews_rival = round(avg_ep_rews_rival, 2)
        avg_actor_loss = round(avg_actor_loss, 5)

        self.logger['avg_batch_losses'].append(avg_actor_loss)

        self.logger['avg_batch_rewards'].append(avg_ep_rews)
        self.logger['avg_batch_rewards_rival'].append(avg_ep_rews_rival)

        loss_array = self.logger['avg_batch_losses'][-50:]
        avg_50_loss = sum(loss_array) / len(loss_array)
        avg_50_loss = round(avg_50_loss, 5)
        min_50_loss = min(loss_array)
        max_50_loss = max(loss_array)

        reward_array = avg_ep_rews_last_50
        avg_50_reward = sum(reward_array) / len(reward_array)
        avg_50_reward = round(avg_50_reward, 2)
        min_50_reward = min(reward_array)
        max_50_reward = max(reward_array)

        reward_array_rival = avg_ep_rews_rival_last_50
        avg_50_reward_rival = sum(reward_array_rival) / len(reward_array_rival)
        avg_50_reward_rival = round(avg_50_reward_rival, 2)
        min_50_reward_rival = min(reward_array_rival)
        max_50_reward_rival = max(reward_array_rival)

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Episodic Return Rival: {avg_ep_rews_rival}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)
        print("[Avg,Max,Min Loss]", avg_50_loss, max_50_loss, min_50_loss)
        print("[Avg,Max,Min Reward]", avg_50_reward, max_50_reward, min_50_reward)
        print("Rival [Avg,Max,Min Reward]", avg_50_reward_rival, max_50_reward_rival, min_50_reward_rival)
        print("List Array Length Loss, Reward", len(loss_array), len(reward_array))
        print("wins loses draws", self.logger['batch_win'][0], self.logger['batch_lose'][0], self.logger['batch_draw'][0])
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        writer.add_scalar("general/AVG_episode_rewards", float(avg_ep_rews), i_so_far)
        writer.add_scalar("general/AVG_rival_episode_rewards", float(avg_ep_rews_rival), i_so_far)
        writer.add_scalar("general/AVG_Loss", float(avg_actor_loss), i_so_far)

        writer.add_scalar("res/number_of_wins", int(self.logger['batch_win'][0]), i_so_far)
        writer.add_scalar("res/number_of_loses", int(self.logger['batch_lose'][0]), i_so_far)
        writer.add_scalar("res/number_of_draws", int(self.logger['batch_draw'][0]), i_so_far)

        writer.add_scalar("last_50_episode/AVG_Reward", float(avg_50_reward), i_so_far)
        writer.add_scalar("last_50_episode/AVG_Reward_rival", float(avg_50_reward_rival), i_so_far)

        writer.add_scalar("last_50_episode/MIN_Reward", float(min_50_reward), i_so_far)
        writer.add_scalar("last_50_episode/MIN_Reward_rival", float(min_50_reward_rival), i_so_far)

        writer.add_scalar("last_50_episode/MAX_Reward", float(max_50_reward), i_so_far)
        writer.add_scalar("last_50_episode/MAX_Reward_rival", float(max_50_reward_rival), i_so_far)

        writer.add_scalar("general/avg_ep_lens", float(avg_ep_lens), i_so_far)

        writer.add_scalar("delta_t/Episode", float(delta_t), i_so_far)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['batch_rews_rival'] = []
        self.logger['actor_losses'] = []
