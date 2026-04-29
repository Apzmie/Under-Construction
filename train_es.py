from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, action_scale=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)

        self.init_weights(self.fc1.weight, self.fc1.bias)
        self.init_weights(self.fc2.weight, self.fc2.bias)
        self.init_weights(self.mean.weight, self.mean.bias)
        
        self.action_scale = action_scale

    def init_weights(self, weight, bias):
        nn.init.normal_(weight, mean=0, std=0.1)
        nn.init.zeros_(bias)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale
        mean = torch.clamp(mean, -1, 1)
        return mean


class ESAgent:
    def __init__(self, state_dim, action_dim, sigma=0.02, lr=0.005, population=24):
        self.model = Actor(state_dim, action_dim)
        self.sigma = sigma
        self.lr = lr
        self.population = population
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.tmp_model = Actor(state_dim, action_dim)

    def get_weights(self):
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def set_weights(self, model, weights):
        pos = 0
        for p in model.parameters():
            size = p.data.numel()
            p.data.copy_(weights[pos:pos+size].view(p.size()))
            pos += size

    def sample_population(self, base_weights):
        noise_list = []
        weight_list = []

        half_pop = self.population // 2
        for _ in range(half_pop):
            noise = torch.randn_like(base_weights)
            pos_weights = base_weights + self.sigma * noise
            neg_weights = base_weights - self.sigma * noise
            noise_list.append(noise)
            weight_list.append(pos_weights)
            weight_list.append(neg_weights)
        return noise_list, weight_list

    def update(self, rewards, noise_list, base_weights):                       
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        direction = torch.zeros_like(base_weights)

        for i, noise in enumerate(noise_list):
            r_pos = normalized_rewards[2*i]
            r_neg = normalized_rewards[2*i + 1]
            direction += (r_pos - r_neg) * noise

        pseudo_gradient = direction / (self.population * self.sigma)
        pos = 0
        self.optimizer.zero_grad()
        for p in self.model.parameters():
            size = p.data.numel()
            p.grad = -pseudo_gradient[pos:pos+size].view(p.size()).clone()
            pos += size

        self.optimizer.step()
        return pseudo_gradient


if __name__ == "__main__":
    channel1 = EngineConfigurationChannel()
    channel1.set_configuration_parameters(time_scale=20.0)
    channel2 = EngineConfigurationChannel()
    channel2.set_configuration_parameters(time_scale=20.0)
    env = UnityEnvironment(file_name="Build.x86_64", side_channels=[channel1], no_graphics=True, worker_id=0)
    test_env = UnityEnvironment(file_name="Build.x86_64", side_channels=[channel2], no_graphics=True, worker_id=1)
    env.reset()
    test_env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    t_behavior_name = list(test_env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    state_dim = spec.observation_specs[0].shape[0]
    action_dim = spec.action_spec.continuous_size
    agent = ESAgent(state_dim, action_dim)
    #agent.model.load_state_dict(torch.load("saved_model.pth"))
    writer = SummaryWriter(log_dir="a/")

    test_interval = 1
    max_step = 1000

    update_count = 0
    best_test_score = -float('inf')

    while True:
        max_step_count = 0

        base_weights = agent.get_weights()
        noise_list, weight_list = agent.sample_population(base_weights)    #

        total_rewards = np.zeros(agent.population)
        episode_dones = np.zeros(agent.population, dtype=bool)

        env.reset()
        decision_steps, _ = env.get_steps(behavior_name)
        id_to_index = {agent_id: i for i, agent_id in enumerate(decision_steps.agent_id)}

        while not np.all(episode_dones):
            if len(decision_steps) > 0:
                states = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)
                actions = np.zeros((len(states), action_dim))

                with torch.no_grad():
                    for j, agent_id in enumerate(decision_steps.agent_id):
                        i = id_to_index[agent_id]                                             
                        if episode_dones[i]:
                            actions[j] = np.zeros(action_dim)
                            continue
                            
                        agent.set_weights(agent.tmp_model, weight_list[i])    #
                        #agent.set_weights(agent.tmp_model, base_weights)    #
                        agent.tmp_model.action_scale = agent.model.action_scale
                        action = agent.tmp_model(states[j])
                        actions[j] = action.numpy()

                env.set_actions(behavior_name, ActionTuple(continuous=actions))

            env.step()
            max_step_count += 1
            if max_step_count >= max_step:
                episode_dones[:] = True
                break

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            for j, agent_id in enumerate(terminal_steps.agent_id):
                i = id_to_index[agent_id]
                if not episode_dones[i]:
                    total_rewards[i] += terminal_steps.reward[j]
                    episode_dones[i] = True

            for j, agent_id in enumerate(decision_steps.agent_id):
                i = id_to_index[agent_id]
                if not episode_dones[i]:
                    total_rewards[i] += decision_steps.reward[j]

        pseudo_gradient = agent.update(torch.tensor(total_rewards, dtype=torch.float32), noise_list, base_weights)
        update_count += 1
        #agent.model.action_scale = max(1.0, agent.model.action_scale * 0.9)    # 5 -> 16
        #max_step = max(200, max_step_count * 2)

        avg_reward = np.mean(total_rewards)
        writer.add_scalar("Train/Average_Reward", avg_reward, update_count)

        grad_norm = pseudo_gradient.norm()
        writer.add_scalar("Train/Gradient_Norm", grad_norm, update_count)

        if update_count % test_interval == 0:
            print(f"Update Count {update_count}")
            test_env.reset()
            t_decision_steps, _ = test_env.get_steps(t_behavior_name)
            n_test_agents = len(t_decision_steps.agent_id)
            test_rewards = np.zeros(n_test_agents)
            test_episode_dones = np.zeros(n_test_agents, dtype=bool)
            test_id_to_index = {agent_id: i for i, agent_id in enumerate(t_decision_steps.agent_id)}
            test_max_step_count = 0
            
            while not np.all(test_episode_dones) and test_max_step_count < max_step:
                if len(t_decision_steps) > 0:
                    t_states = torch.from_numpy(t_decision_steps.obs[0]).to(torch.float32)
                    t_actions = np.zeros((len(t_states), action_dim))
                    with torch.no_grad():
                        for j, agent_id in enumerate(t_decision_steps.agent_id):
                            i = test_id_to_index[agent_id]
                            if test_episode_dones[i]:
                                t_actions[j] = np.zeros(action_dim)
                                continue
                                
                            t_action = agent.model(t_states[j])
                            t_actions[j] = t_action.numpy()
                            
                    test_env.set_actions(t_behavior_name, ActionTuple(continuous=t_actions))

                test_env.step()
                test_max_step_count += 1
                t_decision_steps, t_terminal_steps = test_env.get_steps(t_behavior_name)

                for j, agent_id in enumerate(t_terminal_steps.agent_id):
                    i = test_id_to_index[agent_id]
                    if not test_episode_dones[i]:
                        test_rewards[i] += t_terminal_steps.reward[j]
                        test_episode_dones[i] = True

                for j, agent_id in enumerate(t_decision_steps.agent_id):
                    i = test_id_to_index[agent_id]
                    if not test_episode_dones[i]:
                        test_rewards[i] += t_decision_steps.reward[j]

            test_average_reward = np.mean(test_rewards)
            test_rewards_std = np.std(test_rewards)
            stability_score = test_average_reward - test_rewards_std
            writer.add_scalar("Test/Average_Reward", test_average_reward, update_count)
            writer.add_scalar("Test/Stability_Score", stability_score, update_count)
            writer.add_scalar("Test/Min_Reward", np.min(test_rewards), update_count)
            print(f"[Test] {stability_score:.4f}")

            if stability_score > best_test_score:
                best_test_score = stability_score
                torch.save(agent.model.state_dict(), "best_model.pth")
                print(f"[Test] Model saved as 'best_model.pth' at new best score {best_test_score:.4f}")

