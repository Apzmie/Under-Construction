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


class NextStateModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.posterior_mean = nn.Linear(hidden_dim + state_dim, state_dim)
        self.posterior_log_std = nn.Linear(hidden_dim + state_dim, state_dim)
        self.prior_mean = nn.Linear(hidden_dim, state_dim)
        self.prior_log_std = nn.Linear(hidden_dim, state_dim)
    
    def observe(self, state, action, next_state):
        x = torch.cat([state, action], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        
        posterior_input = torch.cat([x, next_state], dim=-1)               
        posterior_mean = self.posterior_mean(posterior_input)
        posterior_log_std = torch.clamp(self.posterior_log_std(posterior_input), -5, 2)
        posterior_std = torch.exp(posterior_log_std)
        posterior_dist = torch.distributions.Normal(posterior_mean, posterior_std)
        posterior_next_state = posterior_dist.rsample()

        prior_mean = self.prior_mean(x)
        prior_log_std = torch.clamp(self.prior_log_std(x), -5, 2)
        prior_std = torch.exp(prior_log_std)
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)
        prior_next_state = prior_dist.rsample()      
                
        return posterior_dist, posterior_next_state, prior_dist, prior_next_state
        
    def imagine(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        prior_mean = self.prior_mean(x)
        prior_log_std = torch.clamp(self.prior_log_std(x), -5, 2)
        prior_std = torch.exp(prior_log_std)
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)
        prior_next_state = prior_dist.rsample()         
        return prior_next_state          
        

class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        
        nn.init.zeros_(self.reward.weight)
        nn.init.zeros_(self.reward.bias)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        reward = self.reward(x)
        return reward
        

class ContinueModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.continue_logit = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        logit = self.continue_logit(x)
        return logit
        
        
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.next_state_model = NextStateModel(state_dim, action_dim)        
        self.reward_model = RewardModel(state_dim, action_dim)
        self.continue_model = ContinueModel(state_dim, action_dim)    
    
    def loss(self, state, action, next_state, reward, continuation):
        posterior_dist, posterior_next_state, prior_dist, prior_next_state = self.next_state_model.observe(state, action, next_state)
        pred_reward = self.reward_model(state, action)
        pred_continue_logit = self.continue_model(state, action)
        
        #==========================================
        
        next_state_loss = F.mse_loss(posterior_next_state, next_state)
        
        #==========================================
       
        reward_loss = F.mse_loss(pred_reward, reward)
        
        #==========================================
        
        continue_loss = F.binary_cross_entropy_with_logits(pred_continue_logit, continuation)
        
        #==========================================
        
        posterior_dist_detached = torch.distributions.Normal(
            posterior_dist.loc.detach(),
            posterior_dist.scale.detach()
        )
        
        prior_dist_detached = torch.distributions.Normal(
            prior_dist.loc.detach(),
            prior_dist.scale.detach()
        )
        
        dyn_loss = torch.distributions.kl_divergence(
            posterior_dist_detached,
            prior_dist
        ).mean()
        
        rep_loss = torch.distributions.kl_divergence(
            posterior_dist,
            prior_dist_detached
        ).mean()
        
        dyn_loss = torch.clamp(dyn_loss, min=1.0)
        rep_loss = torch.clamp(rep_loss, min=1.0)
        
        dist_loss = dyn_loss + 0.1 * rep_loss 
        
        #==========================================
        
        total_loss = next_state_loss + reward_loss + continue_loss + dist_loss
        return total_loss, next_state_loss, reward_loss, continue_loss, dist_loss

        
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
        
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value
        
        
class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, lr=3e-4):
        super().__init__()
        self.world_model = WorldModel(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor = Actor(state_dim, action_dim)

        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr)        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        #==========================================
        ### Load Actor (fc1, fc2, mean) ###
        
        state_dict = torch.load(f"{BASE_DIR}/previous_model.pth")
        self.actor.fc1.load_state_dict({"weight": state_dict["fc1.weight"], "bias": state_dict["fc1.bias"]})
        self.actor.fc2.load_state_dict({"weight": state_dict["fc2.weight"], "bias": state_dict["fc2.bias"]})
        self.actor.mean.load_state_dict({"weight": state_dict["mean.weight"], "bias": state_dict["mean.bias"]})
        
        with torch.no_grad():        
            self.actor.log_std.weight.zero_()
            self.actor.log_std.bias.fill_(-2)        
        
        #==========================================
        
    def imagine(self, state, horizon=5):
        imagined_states = []
        imagined_actions = []
        imagined_rewards = []
        imagined_next_states = []
        imagined_continuations = []
        imagined_log_probs = []
        
        for t in range(horizon):
            action, log_prob = self.actor.sample(state)
            
            next_state = self.world_model.next_state_model.imagine(state, action) 
            reward = self.world_model.reward_model(state, action)
            continue_logit = self.world_model.continue_model(state, action)
            continuation = torch.sigmoid(continue_logit)          
            
            imagined_states.append(state)
            imagined_actions.append(action)
            imagined_rewards.append(reward)
            imagined_next_states.append(next_state)
            imagined_continuations.append(continuation)
            imagined_log_probs.append(log_prob)
            
            state = next_state
        
        imagined_states = torch.stack(imagined_states, dim=1)    
        imagined_actions = torch.stack(imagined_actions, dim=1)
        imagined_rewards = torch.stack(imagined_rewards, dim=1)
        imagined_next_states = torch.stack(imagined_next_states, dim=1)
        imagined_continuations = torch.stack(imagined_continuations, dim=1)
        imagined_log_probs = torch.stack(imagined_log_probs, dim=1)
       
        return imagined_states, imagined_actions, imagined_rewards, imagined_next_states, imagined_continuations, imagined_log_probs
    
    def compute_return(self, rewards, next_values, continuations, gamma=0.99, lambda_=0.95):
        B, H, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        next_returns = next_values[:, -1, :]
        
        for t in reversed(range(H)):    
            next_returns = rewards[:, t, :] + gamma * continuations[:, t, :] * ((1 - lambda_) * next_values[:, t, :] + lambda_ * next_returns)            
            returns[:, t, :] = next_returns
            
        return returns  
                
    def world_model_update(self, batch):
        state = torch.FloatTensor(batch['state'])
        action = torch.FloatTensor(batch['action'])
        reward = torch.FloatTensor(batch['reward'])
        next_state = torch.FloatTensor(batch['next_state'])
        done = torch.FloatTensor(batch['done'])
        continuation = 1.0 - done
        
        world_model_loss, next_state_loss, reward_loss, continue_loss, dist_loss = self.world_model.loss(state, action, next_state, reward, continuation)
        self.world_model_optimizer.zero_grad()
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()       
        
        return {
            "world_model_loss": world_model_loss.item(),
            "next_state_loss": next_state_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
            "dist_loss": dist_loss.item(),
        }
        
    def update(self, batch):
        state = torch.FloatTensor(batch['state'])
        action = torch.FloatTensor(batch['action'])
        reward = torch.FloatTensor(batch['reward'])
        next_state = torch.FloatTensor(batch['next_state'])
        done = torch.FloatTensor(batch['done'])
        continuation = 1.0 - done
        
        #==========================================
        
        for p in self.world_model.parameters():
            p.requires_grad = False 
        for p in self.actor.parameters():
            p.requires_grad = False 
            
        imagined_states, imagined_actions, imagined_rewards, imagined_next_states, imagined_continuations, imagined_log_probs = self.imagine(state)
        imagined_values = self.critic(imagined_states)
        with torch.no_grad():
            imagined_next_values = self.critic(imagined_next_states)
            returns = self.compute_return(imagined_rewards, imagined_next_values, imagined_continuations)
        
        critic_loss = F.mse_loss(imagined_values, returns)        
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
            
        imagined_states, imagined_actions, imagined_rewards, imagined_next_states, imagined_continuations, imagined_log_probs = self.imagine(state)
        imagined_next_values = self.critic(imagined_next_states)
        returns = self.compute_return(imagined_rewards, imagined_next_values, imagined_continuations)
        
        actor_loss = -returns.mean() + 0.01 * imagined_log_probs.mean()        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for p in self.world_model.parameters():
            p.requires_grad = True
        for p in self.critic.parameters():
            p.requires_grad = True        

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
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
    buffer = ReplayBuffer(state_dim, action_dim)
    writer = SummaryWriter(log_dir=BASE_DIR)
    
    # Set random_exploration_steps, learning_starts to 0
    #load_checkpoint(f"{BASE_DIR}/checkpoint.pth", agent, buffer)
    
    random_exploration_steps = 1000
    learning_starts = 500
    test_interval = 1000
    test_max_step = 1000
    
    total_steps = 0
    update_count = 0
    save_idx = 0
    best_test_reward = -float('inf')
    world_model_steps = 0
    
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
            wm_metrics = agent.world_model_update(batch)
            world_model_steps += 1

            if world_model_steps >= 10000:
                batch = buffer.sample()
                metrics = agent.update(batch) 
                update_count += 1
                for k, v in wm_metrics.items():
                    writer.add_scalar(f"Train/{k}", v, update_count)            
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
                     
