from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter       

BASE_DIR = ""


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        
        self.next_state = nn.Linear(hidden_dim, state_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        
        next_state = self.next_state(x)
        reward = self.reward(x)
        
        return next_state, reward
        

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))       
        mean = self.mean(x)        
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()        
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob
        
    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)
        

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value
        
        
class Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.world_model = WorldModel(state_dim, action_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr)
        
    def update_world_model(self, state, action, next_state, reward):
        pred_next_state, pred_reward = self.world_model(state, action)
        
        recon_loss = F.mse_loss(next_state, pred_next_state)
        reward_loss = F.mse_loss(reward, pred_reward) 
        
        total_loss = recon_loss + reward_loss
        return total_loss, recon_loss, reward_loss
        
    def imagine(self, state, horizon=15):
        imagined_states = []
        imagined_actions = []
        imagined_rewards = []
        imagined_next_states = []
        imagined_values = []
        imagined_next_values = []
        
        for _ in range(horizon):
            action, _ = self.actor.sample(state)
            pred_next_state, pred_reward = self.world_model(state, action)
            value = self.critic(state)
            next_value = self.critic(pred_next_state)            
            
            imagined_states.append(state)
            imagined_actions.append(action)
            imagined_rewards.append(pred_reward)
            imagined_next_states.append(pred_next_state)
            imagined_values.append(value)
            imagined_next_values.append(next_value)

            state = pred_next_state
            
        imagined_states = torch.stack(imagined_states, dim=1)
        imagined_actions = torch.stack(imagined_actions, dim=1)
        imagined_rewards = torch.stack(imagined_rewards, dim=1)
        imagined_next_states = torch.stack(imagined_next_states, dim=1)
        imagined_values = torch.stack(imagined_values, dim=1)
        imagined_next_values = torch.stack(imagined_next_values, dim=1)
        
        return imagined_states, imagined_actions, imagined_rewards, imagined_next_states, imagined_values, imagined_next_values
        
    def compute_return(self, rewards, next_values, gamma=0.99, lambda_=0.95):
        B, H, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        next_returns = next_values[:, -1, :]
        
        for t in reversed(range(H)):    
            next_returns = rewards[:, t, :] + gamma * ((1 - lambda_) * next_values[:, t, :] + lambda_ * next_returns)            
            returns[:, t, :] = next_returns
            
        return returns               
    
    def critic_loss(self, values, returns):
        critic_loss = F.mse_loss(values, returns.detach())     
        return critic_loss
        
    def actor_loss(self, returns):          
        actor_loss = -returns.mean()       
        return actor_loss

    def update(self, batch):
        state = torch.FloatTensor(batch['state'])
        action = torch.FloatTensor(batch['action'])
        reward = torch.FloatTensor(batch['reward'])
        next_state = torch.FloatTensor(batch['next_state'])
        done = torch.FloatTensor(batch['done'])
        
        #==========================================
        
        world_model_loss, recon_loss, reward_loss = self.update_world_model(state, action, next_state, reward)
        self.world_model_optimizer.zero_grad()
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()
        
        #==========================================
        
        for p in self.world_model.parameters():
            p.requires_grad = False 
        for p in self.actor.parameters():
            p.requires_grad = False 
        
        with torch.no_grad():
            imagined_states, imagined_actions, imagined_rewards, imagined_next_states, imagined_values, imagined_next_values = self.imagine(state)
            returns = self.compute_return(imagined_rewards, imagined_next_values)
        
        imagined_values = self.critic(imagined_states)
        
        critic_loss = self.critic_loss(imagined_values, returns)        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        for p in self.world_model.parameters():
            p.requires_grad = True 
        for p in self.actor.parameters():
            p.requires_grad = True 
        
        #==========================================
        
        for p in self.world_model.parameters():
            p.requires_grad = False
        for p in self.critic.parameters():
            p.requires_grad = False      
        
        imagined_states, imagined_actions, imagined_rewards, imagined_next_states, imagined_values, imagined_next_values = self.imagine(state)
        returns = self.compute_return(imagined_rewards, imagined_next_values)
        
        actor_loss = self.actor_loss(returns)        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        for p in self.world_model.parameters():
            p.requires_grad = True
        for p in self.critic.parameters():
            p.requires_grad = True
        
        return {
            "world_model_loss": world_model_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),          
        }
        
        
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), batch_size=256):
        self.max_size = max_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idx = np.random.randint(0, self.size, size=self.batch_size)

        return {
            "state": self.state[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_state": self.next_state[idx],
            "done": self.done[idx],
        }
    

if __name__ == "__main__":
    channel1 = EngineConfigurationChannel()
    channel1.set_configuration_parameters(time_scale=20.0)
    channel2 = EngineConfigurationChannel()
    channel2.set_configuration_parameters(time_scale=20.0)
    env = UnityEnvironment(file_name=f"{BASE_DIR}/Build.x86_64", side_channels=[channel1], no_graphics=True, worker_id=0)
    test_env = UnityEnvironment(file_name=f"{BASE_DIR}/Build.x86_64", side_channels=[channel2], no_graphics=True, worker_id=1)
    env.reset()
    test_env.reset()
    
    behavior_name = list(env.behavior_specs.keys())[0]
    t_behavior_name = list(test_env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    state_dim = spec.observation_specs[0].shape[0]
    action_dim = spec.action_spec.continuous_size
    agent = Agent(state_dim, action_dim)
    #agent.actor.load_state_dict(torch.load(f"{BASE_DIR}/period_model.pth"))
    buffer = ReplayBuffer(state_dim, action_dim)
    writer = SummaryWriter(log_dir=BASE_DIR)
    
    random_exploration_steps = 1000
    learning_starts = 500
    test_interval = 1000
    test_max_step = 1000
    
    total_steps = 0
    update_count = 0
    save_idx = 0
    best_test_reward = -float('inf')
    
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        agent_ids = decision_steps.agent_id
        if len(agent_ids) > 0:
            states_tensor = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)  
            
            if total_steps < random_exploration_steps:
                actions = np.random.uniform(low=-1.0, high=1.0, size=(len(agent_ids), action_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    actions, _ = agent.actor.sample(states_tensor)   
                actions = actions.cpu().numpy().astype(np.float32)
                
            env.set_actions(behavior_name, ActionTuple(continuous=actions))
            
        env.step()
        next_decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        for i, agent_id in enumerate(agent_ids):
            state = states_tensor[i].cpu().numpy()
            action = actions[i]

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
                
            buffer.add(state, action, reward, next_state, done)
            total_steps += 1
         
        if total_steps >= learning_starts:
             batch = buffer.sample()
             metrics = agent.update(batch) 
             update_count += 1           
             for k, v in metrics.items():
                 writer.add_scalar(f"Train/{k}", v, update_count)               
             
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
                         t_states_tensor = torch.from_numpy(t_decision_steps.obs[0]).to(torch.float32)                        
                         with torch.no_grad():
                             t_actions = agent.actor.deterministic(t_states_tensor)                    
                         t_actions = t_actions.cpu().numpy().astype(np.float32)
                         
                         for j, agent_id in enumerate(t_agent_ids):
                             idx = test_id_to_index[agent_id]
                             if test_episode_dones[idx]:
                                 t_actions[j] = np.zeros(action_dim)
                                
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
                 writer.add_scalar("Test/Average_Reward", test_average_reward, update_count)
                 print(f"{test_average_reward:.4f}")
                 torch.save(agent.actor.state_dict(), f"{BASE_DIR}/period_model.pth")
                 #save_checkpoint(f"{BASE_DIR}/checkpoint.pth", agent, buffer)                    
                         
                 if test_average_reward > best_test_reward:
                     best_test_reward = test_average_reward
                     save_idx += 1
                     torch.save(agent.actor.state_dict(), f"{BASE_DIR}/#({save_idx})best_{best_test_reward:.4f}.pth") 
                     print(f"[Test] Model saved at new best reward {best_test_reward:.4f}")
                     
