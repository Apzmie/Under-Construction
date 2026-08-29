from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 

BASE_DIR = ""


class Encoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, embed_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.embed = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, state):
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        embed = self.embed(x)
        return embed
        

class RSSM(nn.Module):
    def __init__(self, action_dim, latent_dim=32, hidden_dim=128, embed_dim=64):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)        
        self.posterior_mean = nn.Linear(hidden_dim + embed_dim, latent_dim)
        self.posterior_std = nn.Linear(hidden_dim + embed_dim, latent_dim)        
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_std = nn.Linear(hidden_dim, latent_dim)         
        
    def observe(self, latent, action, memory, embed):
        gru_input = torch.cat([latent, action], dim=-1)
        memory = self.gru(gru_input, memory)
        
        posterior_input = torch.cat([memory, embed], dim=-1)
        posterior_mean = self.posterior_mean(posterior_input)
        posterior_std = F.softplus(self.posterior_std(posterior_input)) + 1e-4
        posterior_dist = torch.distributions.Normal(posterior_mean, posterior_std)
        posterior_latent = posterior_dist.rsample()
        
        prior_mean = self.prior_mean(memory)
        prior_std = F.softplus(self.prior_std(memory)) + 1e-4
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)
        prior_latent = prior_dist.rsample()
        
        return memory, posterior_dist, posterior_latent, prior_dist, prior_latent
        
    def imagine(self, latent, action, memory):
        gru_input = torch.cat([latent, action], dim=-1)
        memory = self.gru(gru_input, memory)
        
        prior_mean = self.prior_mean(memory)
        prior_std = F.softplus(self.prior_std(memory)) + 1e-4
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)
        prior_latent = prior_dist.rsample()
        
        return memory, prior_latent

        
class Decoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.recon = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)        
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        recon = self.recon(x)
        return recon
        
        
class RewardModel(nn.Module):
    def __init__(self, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        reward = self.reward(x)
        return reward
        

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, memory_dim=128, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(state_dim)
        self.rssm = RSSM(action_dim)
        self.decoder = Decoder(state_dim)
        self.reward_model = RewardModel()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=6e-4)
        
        self.memory_dim = memory_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
    def recon_loss(self, memory, posterior_latent, state):
        recon = self.decoder(memory, posterior_latent)
        recon_loss = F.mse_loss(state, recon)  
        return recon_loss
        
    def reward_loss(self, memory, posterior_latent, reward):
        pred_reward = self.reward_model(memory, posterior_latent)
        reward_loss = F.mse_loss(reward, pred_reward)
        return reward_loss
        
    def dist_loss(self, posterior_dist, prior_dist):
        dist_loss = torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()
        dist_loss = torch.clamp(dist_loss, min=3.0)
        return dist_loss  
    
    def update(self, episodes, starts, states, actions, rewards, next_states, dones):
        batch_size = states.shape[0]
        sequence_length = states.shape[1]
        
        total_recon_loss = 0.0
        total_reward_loss = 0.0
        total_dist_loss = 0.0
        
        memories = []
        latents = []
        
        with torch.no_grad():
            for i in range(batch_size):
                initial_memory = torch.zeros(1, self.memory_dim)
                initial_latent = torch.zeros(1, self.latent_dim)
                initial_action = torch.zeros(1, self.action_dim)
                
                state = episodes[i][0]["state"].unsqueeze(0)
                embed = self.encoder(state)
                memory, _, latent, _, _ = self.rssm.observe(initial_latent, initial_action, initial_memory, embed)                 
                
                for t in range(starts[i]):
                    action = episodes[i][t]["action"].unsqueeze(0)
                    next_state = episodes[i][t]["next_state"].unsqueeze(0)
                    embed = self.encoder(next_state)
                    memory, _, latent, _, _ = self.rssm.observe(latent, action, memory, embed)                  
                
                memories.append(memory.squeeze(0))
                latents.append(latent.squeeze(0))
                
        memory = torch.stack(memories)
        latent = torch.stack(latents)
        
        for t in range(sequence_length):
            action = actions[:, t, :]
            reward = rewards[:, t].unsqueeze(-1)
            next_state = next_states[:, t, :]
            
            embed = self.encoder(next_state)
            memory, posterior_dist, posterior_latent, prior_dist, prior_latent = self.rssm.observe(latent, action, memory, embed)
        
            recon_loss = self.recon_loss(memory, posterior_latent, next_state)
            reward_loss = self.reward_loss(memory, posterior_latent, reward)
            dist_loss = self.dist_loss(posterior_dist, prior_dist)
        
            total_recon_loss += recon_loss
            total_reward_loss += reward_loss
            total_dist_loss += dist_loss
            
            latent = posterior_latent
            
        total_recon_loss /= sequence_length
        total_reward_loss /= sequence_length
        total_dist_loss /= sequence_length
        
        total_loss = total_recon_loss + total_reward_loss + total_dist_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100.0)
        self.optimizer.step()

        return total_loss.item(), total_recon_loss.item(), total_reward_loss.item(), total_dist_loss.item(), memory.detach(), latent.detach()   
                           
        
class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        mean = self.mean(x)
        std = F.softplus(self.std(x)) + 1e-4                
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        action = torch.tanh(action)
        return action
        
    def deterministic(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        mean = self.mean(x)
        action = torch.tanh(mean)
        return action
                
class Critic(nn.Module):
    def __init__(self, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        value = self.value(x)
        return value


class Agent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.world_model = WorldModel(state_dim, action_dim)
        self.actor = Actor(action_dim)
        self.critic = Critic()
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=8e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=8e-5)

    def imagine_with_AC(self, memory, latent, horizon=15):
        pred_values = []
        pred_rewards = []
        
        for t in range(horizon):
            action = self.actor(memory, latent)
            memory, prior_latent = self.world_model.rssm.imagine(latent, action, memory)

            pred_value = self.critic(memory, prior_latent)
            pred_reward = self.world_model.reward_model(memory, prior_latent)
            
            pred_values.append(pred_value)
            pred_rewards.append(pred_reward)
            
            latent = prior_latent
            
        action = self.actor(memory, latent)
        memory, prior_latent = self.world_model.rssm.imagine(latent, action, memory)
        pred_value = self.critic(memory, prior_latent)
        pred_values.append(pred_value)   
        
        pred_values = torch.stack(pred_values, dim=1)
        pred_rewards = torch.stack(pred_rewards, dim=1)
        
        return pred_values, pred_rewards
    
    def compute_return(self, rewards, values, gamma=0.99, lambda_=0.95):
        B, H, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        next_returns = values[:, -1, :]
        
        for t in reversed(range(H)):
            if t == H-1:
                next_values = values[:, -1, :]
            else:
                next_values = values[:, t+1, :]
                
            next_returns = rewards[:, t, :] + gamma * ((1 - lambda_) * next_values + lambda_ * next_returns)            
            returns[:, t, :] = next_returns
            
        return returns    
    
    def actor_loss(self, returns):
        actor_loss = -returns.mean()
        return actor_loss
        
    def critic_loss(self, pred_values, returns):
        critic_loss = F.mse_loss(pred_values, returns.detach())
        return critic_loss
    
    def update(self, memory, latent):
        for param in self.world_model.parameters():
            param.requires_grad = False
        
        pred_values, pred_rewards = self.imagine_with_AC(memory, latent)
        returns = self.compute_return(pred_rewards, pred_values)     
        
        for param in self.critic.parameters():
            param.requires_grad = False
            
        actor_loss = self.actor_loss(returns)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()
        
        for param in self.critic.parameters():
            param.requires_grad = True        
        
        for param in self.actor.parameters():
            param.requires_grad = False
            
        pred_values, pred_rewards = self.imagine_with_AC(memory, latent)
        returns = self.compute_return(pred_rewards, pred_values) 
                
        critic_loss = self.critic_loss(pred_values[:, :-1, :], returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()
        
        for param in self.actor.parameters():
            param.requires_grad = True
        
        for param in self.world_model.parameters():
            param.requires_grad = True
            
        return actor_loss.item(), critic_loss.item()
        

class ReplayBuffer:
    def __init__(self, capacity=1000, batch_size=50, sequence_length=50):
        self.episodes = []
        self.capacity = capacity
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
    def update_sequence_length(self):
        max_length = max(len(episode) for episode in self.episodes)
        if max_length >= 100:
            self.sequence_length = 50
        elif max_length >= 80:
            self.sequence_length = 40
        elif max_length >= 60:
            self.sequence_length = 30
        elif max_length >= 40:
            self.sequence_length = 20
        elif max_length >= 20:
            self.sequence_length = 10
        else:
            self.sequence_length = 1
        
    def add_episode(self, episode):
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)            
        self.episodes.append(episode)
        
    def sample(self):
        self.update_sequence_length()
        
        episodes = []
        starts = []
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        valid_episodes = [episode for episode in self.episodes if len(episode) >= self.sequence_length]
        for _ in range(self.batch_size):
            episode = valid_episodes[np.random.randint(len(valid_episodes))]        
            start = np.random.randint(
                0,
                len(episode) - self.sequence_length + 1
            )
            sequence = episode[start:start + self.sequence_length]                     
            
            episodes.append(episode)
            starts.append(start)
            
            states.append(torch.stack([transition["state"] for transition in sequence]))
            actions.append(torch.stack([transition["action"] for transition in sequence]))
            rewards.append(torch.stack([transition["reward"] for transition in sequence]))
            next_states.append(torch.stack([transition["next_state"] for transition in sequence]))
            dones.append([transition["done"] for transition in sequence])
            
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)  
        dones = torch.tensor(dones, dtype=torch.float32)
        
        return episodes, starts,states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.episodes)
        
        
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
    #agent.load_state_dict(torch.load(f"{BASE_DIR}/period_model.pth"))
    buffer = ReplayBuffer()
    writer = SummaryWriter(log_dir=BASE_DIR)        
    
    memory_dim, latent_dim = 128, 32
    max_step = 500
    update_interval = 10
    update_iterations = 100
    test_interval = 50
    num_agents = 24
    
    agent_dictionary = {}
    add_episode = 0
    update_count = 0
    best_test_reward = -float('inf')
    save_idx = 0
        
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        if len(agent_ids) > 0:
            states_tensor = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)
            
            actions_for_unity = []
            for i, agent_id in enumerate(agent_ids):
                if agent_id not in agent_dictionary:
                    memory = torch.zeros(1, memory_dim)
                    latent = torch.zeros(1, latent_dim)
                    action = torch.zeros(1, action_dim)
            
                    state = states_tensor[i].unsqueeze(0)
                    embed = agent.world_model.encoder(state)
                    memory, _, latent, _, _ = agent.world_model.rssm.observe(latent, action, memory, embed)                
            
                    agent_dictionary[agent_id] = {
                        "memory": memory.squeeze(0),
                        "latent": latent.squeeze(0),
                        "action": action.squeeze(0),
                        "transitions": []
                    }
                    
                memory = agent_dictionary[agent_id]["memory"].unsqueeze(0)
                latent = agent_dictionary[agent_id]["latent"].unsqueeze(0)
                
                with torch.no_grad():
                    action = agent.actor(memory, latent)
                action = action.squeeze(0)                
                agent_dictionary[agent_id]["action"] = action
                actions_for_unity.append(action.numpy())
                
            actions_for_unity = np.array(actions_for_unity)
            env.set_actions(behavior_name, ActionTuple(continuous=actions_for_unity))
        
        env.step()
        next_decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        for i, agent_id in enumerate(agent_ids):
            if agent_id in next_decision_steps:
                reward = next_decision_steps[agent_id].reward
                next_obs = next_decision_steps[agent_id].obs[0]
                done = False
            elif agent_id in terminal_steps:
                reward = terminal_steps[agent_id].reward
                next_obs = terminal_steps[agent_id].obs[0]
                done = True
            else:
                continue
            
            agent_dictionary[agent_id]["transitions"].append({
                "state": states_tensor[i],
                "action": agent_dictionary[agent_id]["action"],
                "reward": torch.tensor(reward, dtype=torch.float32),
                "next_state": torch.from_numpy(next_obs).float(),
                "done": done
            })
            
            next_state = torch.from_numpy(next_obs).float().unsqueeze(0)
            next_embed = agent.world_model.encoder(next_state)
            
            action = agent_dictionary[agent_id]["action"].unsqueeze(0)
            memory = agent_dictionary[agent_id]["memory"].unsqueeze(0)
            latent = agent_dictionary[agent_id]["latent"].unsqueeze(0)
            
            memory, _, latent, _, _ = agent.world_model.rssm.observe(latent, action, memory, next_embed)                
            
            agent_dictionary[agent_id]["memory"] = memory.squeeze(0)
            agent_dictionary[agent_id]["latent"] = latent.squeeze(0)
            
            if done or len(agent_dictionary[agent_id]["transitions"]) >= max_step:
                episode = agent_dictionary[agent_id]["transitions"]
                buffer.add_episode(episode)
                add_episode += 1                
                if done:
                    del agent_dictionary[agent_id]
                else:
                    agent_dictionary[agent_id]["transitions"] = []           
        
                if add_episode > 0 and add_episode % update_interval == 0:    
                    for _ in range(update_iterations):
                        episodes, starts, states, actions, rewards, next_states, dones = buffer.sample()             
                        total_loss, recon_loss, reward_loss, dist_loss, memory, latent = agent.world_model.update(episodes, starts, states, actions, rewards, next_states, dones)
                        actor_loss, critic_loss = agent.update(memory, latent)
                
                        writer.add_scalar("Train/WorldModel_Total_Loss", total_loss, update_count)
                        writer.add_scalar("Train/WorldModel_Reconstruction_Loss", recon_loss, update_count)
                        writer.add_scalar("Train/WorldModel_Reward_Loss", reward_loss, update_count)
                        writer.add_scalar("Train/WorldModel_Distribution_Loss", dist_loss, update_count)
                        writer.add_scalar("Train/Actor_Loss", actor_loss, update_count)
                        writer.add_scalar("Train/Critic_Loss", critic_loss, update_count)
             
                        update_count += 1
                
                    print(f"Update Count {update_count}")               
                    if update_count % test_interval == 0:
                        test_env.reset()
                        t_agent_dictionary = {}
                        t_step = 0
                        test_rewards = 0
                
                        while t_step < max_step:
                            t_decision_steps, _ = test_env.get_steps(t_behavior_name)    
                            t_agent_ids = t_decision_steps.agent_id
                            if len(t_agent_ids) > 0:
                                t_states_tensor = torch.from_numpy(t_decision_steps.obs[0]).to(torch.float32)
                    
                                t_actions_for_unity = []
                                for i, t_agent_id in enumerate(t_agent_ids):
                                    if t_agent_id not in t_agent_dictionary:
                                        t_memory = torch.zeros(1, memory_dim)
                                        t_latent = torch.zeros(1, latent_dim)
                                        t_action = torch.zeros(1, action_dim)
                            
                                        t_state = t_states_tensor[i].unsqueeze(0)
                                        t_embed = agent.world_model.encoder(t_state)
                                        t_memory, _, t_latent, _, _ = agent.world_model.rssm.observe(t_latent, t_action, t_memory, t_embed)                
                                        t_agent_dictionary[t_agent_id] = {
                                            "t_memory": t_memory.squeeze(0),
                                            "t_latent": t_latent.squeeze(0),
                                            "t_action": t_action.squeeze(0),
                                        }
                            
                                    t_memory = t_agent_dictionary[t_agent_id]["t_memory"].unsqueeze(0)
                                    t_latent = t_agent_dictionary[t_agent_id]["t_latent"].unsqueeze(0)
                                    with torch.no_grad():
                                        t_action = agent.actor.deterministic(t_memory, t_latent)
                                    t_action = t_action.squeeze(0)
                                    t_agent_dictionary[t_agent_id]["t_action"] = t_action
                                    t_actions_for_unity.append(t_action.numpy())
                        
                                t_actions_for_unity = np.array(t_actions_for_unity)
                                test_env.set_actions(t_behavior_name, ActionTuple(continuous=t_actions_for_unity))    
             
                            test_env.step()
                            t_step += 1
                            t_next_decision_steps, t_terminal_steps = test_env.get_steps(t_behavior_name) 
                          
                            for i, t_agent_id in enumerate(t_agent_ids):
                                if t_agent_id in t_next_decision_steps:
                                    t_reward = t_next_decision_steps[t_agent_id].reward
                                    t_next_obs = t_next_decision_steps[t_agent_id].obs[0]
                                    t_done = False
                                elif t_agent_id in t_terminal_steps:
                                    t_reward = t_terminal_steps[t_agent_id].reward
                                    t_next_obs = t_terminal_steps[t_agent_id].obs[0]
                                    t_done = True
                                else:
                                    continue
                            
                                test_rewards += t_reward
            
                                t_next_state = torch.from_numpy(t_next_obs).float().unsqueeze(0)
                                t_next_embed = agent.world_model.encoder(t_next_state)
            
                                t_action = t_agent_dictionary[t_agent_id]["t_action"].unsqueeze(0)
                                t_memory = t_agent_dictionary[t_agent_id]["t_memory"].unsqueeze(0)
                                t_latent = t_agent_dictionary[t_agent_id]["t_latent"].unsqueeze(0)
            
                                t_memory, _, t_latent, _, _ = agent.world_model.rssm.observe(t_latent, t_action, t_memory, t_next_embed)                
            
                                t_agent_dictionary[t_agent_id]["t_memory"] = t_memory.squeeze(0)
                                t_agent_dictionary[t_agent_id]["t_latent"] = t_latent.squeeze(0)
                        
                                if t_done:
                                    del t_agent_dictionary[t_agent_id]
                            
                        test_average_reward = test_rewards / num_agents
                        writer.add_scalar("Test/Average_Reward", test_average_reward, update_count)
                        print(f"[Test] {test_average_reward:.4f}")
                        torch.save({
                            "model": agent.state_dict(),
                            "world_model_optimizer": agent.world_model.optimizer.state_dict(),
                            "actor_optimizer": agent.actor_optimizer.state_dict(),
                            "critic_optimizer": agent.critic_optimizer.state_dict(),
                        }, f"{BASE_DIR}/checkpoint.pth")                 
                        torch.save(agent.state_dict(), f"{BASE_DIR}/period_model.pth")
                 
                        if test_average_reward > best_test_reward:
                            best_test_reward = test_average_reward
                            save_idx += 1 
                            torch.save(agent.state_dict(), f"{BASE_DIR}/#({save_idx})best_{best_test_reward:.4f}.pth")
                            print(f"[Test] Model saved at new best reward {best_test_reward:.4f}")              
        
