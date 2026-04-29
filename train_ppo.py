from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, action_scale=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -2.0)
        self.critic = nn.Linear(hidden_dim, 1)
        
        self.action_scale = action_scale

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale
        mean = torch.clamp(mean, -1, 1)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(x).squeeze(-1)
        return mean, std, value      
                

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.returns = []
        self.advantages = []

    def add(self, state, action, reward, next_state, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.__init__()
        
        
class Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95, freeze_actor=False):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.gamma = gamma
        self.lam = lam
        
        self.actor_critic.load_state_dict(torch.load("saved_model.pth"), strict=False)
        
        if freeze_actor:
            for name, param in self.actor_critic.named_parameters():
                if "critic" not in name:
                    param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.actor_critic.parameters()), 
            lr=lr
        )

    def select_action(self, state, train=True):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            mean, std, value = self.actor_critic(state)
            if train:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            else:
                action = mean
                log_prob = None
        
        action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().numpy(), log_prob, value
        
    def compute_gaes(self, buffer):
        with torch.no_grad():
            rewards = torch.FloatTensor(np.array(buffer.rewards))
            values = torch.FloatTensor(np.array(buffer.values))
            dones = torch.FloatTensor(np.array(buffer.dones))

            if dones[-1] == 1.0:
                next_value = 0
            else:
                last_next_state = torch.FloatTensor(buffer.next_states[-1]).unsqueeze(0)
                _, _, last_value = self.actor_critic(last_next_state)
                next_value = last_value.item()

            gaes = [0] * len(rewards)
            last_gae = 0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gaes[t] = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
                next_value = values[t]
                last_gae = gaes[t]

            gaes = torch.FloatTensor(np.array(gaes))
            buffer.advantages = gaes.view(-1)
            buffer.returns = gaes.view(-1) + values.view(-1)

    def update(self, buffers, epochs=5, batch_size=512):
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        
        for buf in buffers:
            all_states.extend(buf.states)
            all_actions.extend(buf.actions)
            all_old_log_probs.extend(buf.log_probs)
            all_returns.append(buf.returns)
            all_advantages.append(buf.advantages)
            
        states = torch.FloatTensor(np.array(all_states))
        actions = torch.FloatTensor(np.array(all_actions))
        old_log_probs = torch.FloatTensor(np.array(all_old_log_probs))
        returns = torch.cat(all_returns).view(-1)
        advantages = torch.cat(all_advantages).view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                mean, std, values = self.actor_critic(states[idx])
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(actions[idx]).sum(-1)
                entropy = dist.entropy().sum(-1)
                
                ratios = torch.exp(log_probs - old_log_probs[idx])
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.view(-1), returns[idx].view(-1))
                entropy_loss = -entropy.mean()
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
        
        for buf in buffers:
            buf.clear()
            
        return policy_loss.item(), value_loss.item(), entropy_loss.item()


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
    agent = Agent(state_dim, action_dim, freeze_actor=False)
    buffer = RolloutBuffer()
    writer = SummaryWriter(log_dir="a/")
    
    target_transitions = 3072    # all transitions per one update
    test_interval = 10
    test_max_step = 1000

    update_count = 0
    total_transitions = 0
    best_test_score = -float('inf')
    agent_buffers = {}
    completed_buffers = []

    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        agent_ids = decision_steps.agent_id
        if len(agent_ids) > 0:
            for agent_id in agent_ids:
                if agent_id not in agent_buffers:
                    agent_buffers[agent_id] = RolloutBuffer()

            states = np.array([decision_steps[aid].obs[0] for aid in agent_ids])
            actions_array, log_probs_tensor, values_tensor = agent.select_action(states)
            actions_tuple = ActionTuple(continuous=actions_array)
            env.set_actions(behavior_name, actions_tuple)

        env.step()
        next_decision_steps, terminal_steps = env.get_steps(behavior_name)

        for i, agent_id in enumerate(agent_ids):
            state = states[i]
            action = actions_array[i]
            log_prob = log_probs_tensor[i]
            value = values_tensor[i]

            if agent_id in terminal_steps:
                reward = terminal_steps[agent_id].reward
                done = 1.0
                next_state = np.zeros_like(state)
            elif agent_id in next_decision_steps:
                reward = next_decision_steps[agent_id].reward
                done = 0.0
                next_state = next_decision_steps[agent_id].obs[0]
            else:
                continue

            agent_buffers[agent_id].add(state, action, reward, next_state, log_prob.item(), value.item(), done)
            total_transitions += 1

            if done == 1.0:
                agent.compute_gaes(agent_buffers[agent_id])
                completed_buffers.append(agent_buffers[agent_id])
                agent_buffers[agent_id] = RolloutBuffer()

        if total_transitions >= target_transitions:
            for aid, buf in agent_buffers.items():
                if len(buf) > 0:
                    agent.compute_gaes(buf)
                    completed_buffers.append(buf)

            policy_loss, value_loss, entropy_loss = agent.update(completed_buffers)
            writer.add_scalar("Train/Policy_Loss", policy_loss, update_count)
            writer.add_scalar("Train/Value_Loss", value_loss, update_count)
            writer.add_scalar("Train/Entropy_Loss", entropy_loss, update_count)

            completed_buffers.clear()
            total_transitions = 0
            agent_buffers = {}
            update_count += 1           
                
            if update_count % test_interval == 0:
                print(f"Update Count {update_count}")
                test_env.reset()
                t_decision_steps, _ = test_env.get_steps(t_behavior_name)
                n_test_agents = len(t_decision_steps.agent_id)
                test_rewards = np.zeros(n_test_agents)
                test_episode_dones = np.zeros(n_test_agents, dtype=bool)
                test_id_to_index = {agent_id: i for i, agent_id in enumerate(t_decision_steps.agent_id)}
                
                test_max_step_count = 0
                while not np.all(test_episode_dones) and test_max_step_count < test_max_step:
                    t_agent_ids = t_decision_steps.agent_id
        
                    if len(t_agent_ids) > 0:
                        t_states = t_decision_steps.obs[0]
                        t_actions_array, _, _ = agent.select_action(t_states, train=False)
            
                        for j, agent_id in enumerate(t_agent_ids):
                            idx = test_id_to_index[agent_id]
                            if test_episode_dones[idx]:
                                t_actions_array[j] = np.zeros(action_dim)
            
                        test_env.set_actions(t_behavior_name, ActionTuple(continuous=t_actions_array))
                        
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
                print(f"{stability_score:.4f}")
                
                if stability_score > best_test_score:
                    best_test_score = stability_score
                    torch.save(agent.actor_critic.state_dict(), "best_model.pth")
                    print(f"[Test] Model saved as 'best_model.pth' at new best score {best_test_score:.4f}")
