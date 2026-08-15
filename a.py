from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 

BASE_DIR = "/home/psh/Two/a"


class Encoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, state):
        x = self.fc1(state)
        x = F.elu(x)
        embed = self.fc2(x)
        return embed
        
       
class RSSM(nn.Module):
    def __init__(self, action_dim, hidden_dim=256, latent_dim=32, embed_dim=128):
        super().__init__()
        input_dim = latent_dim + action_dim
        
        self.W_forget_i = nn.Linear(input_dim, hidden_dim)
        self.W_forget_m = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_info_i = nn.Linear(input_dim, hidden_dim)
        self.W_info_m = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_update_i = nn.Linear(input_dim, hidden_dim)
        self.W_update_m = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_prior = nn.Linear(hidden_dim, hidden_dim)   
        self.W_prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.W_prior_std = nn.Linear(hidden_dim, latent_dim)  
        
        self.W_posterior = nn.Linear(hidden_dim + embed_dim, hidden_dim)   
        self.W_posterior_mean = nn.Linear(hidden_dim, latent_dim)
        self.W_posterior_std = nn.Linear(hidden_dim, latent_dim)  
        
    def gru(self, latent, action, memory):
        input = torch.cat([latent, action], dim=-1)
        
        input_for_forget = self.W_forget_i(input)
        memory_for_forget = self.W_forget_m(memory)
        
        forget_gate = torch.sigmoid(input_for_forget + memory_for_forget)       
        forgotten_memory = forget_gate * memory
           
        forgotten_memory_info = self.W_info_m(forgotten_memory)                    
        input_info = self.W_info_i(input)               
        summed_info = torch.tanh(forgotten_memory_info + input_info)
        
        input_for_update = self.W_update_i(input)
        memory_for_update = self.W_update_m(memory)

        update_gate = torch.sigmoid(input_for_update + memory_for_update)        
        updated_memory = (1 - update_gate) * memory + update_gate * summed_info
        return updated_memory
        
    def prior(self, memory):
        x = self.W_prior(memory)
        x = F.elu(x)        
        mean = self.W_prior_mean(x)
        std = self.W_prior_std(x)
        std = F.softplus(std) + 1e-4
        
        return mean, std
        
    def posterior(self, memory, embed):
        x = torch.cat([memory, embed], dim=-1)       
        x = self.W_posterior(x)
        x = F.elu(x)        
        mean = self.W_posterior_mean(x)
        std = self.W_posterior_std(x)
        std = F.softplus(std) + 1e-4
        
        return mean, std
        
    def sample(self, mean, std):
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def observe(self, embeds, actions, initial_memory, initial_latent):
        memory = initial_memory
        latent = initial_latent
        
        memories = []
        latents = []

        prior_means = []
        prior_stds = []

        posterior_means = []
        posterior_stds = []
        
        seq_len = embeds.shape[1]  # [B, seq_len, embed_dim]       
        for t in range(seq_len):
            embed = embeds[:, t, :]
            action = actions[:, t, :]
            
            memory = self.gru(latent, action, memory)
            prior_mean, prior_std = self.prior(memory)
            posterior_mean, posterior_std = self.posterior(memory, embed)
            
            latent = self.sample(posterior_mean, posterior_std)
            
            memories.append(memory)
            latents.append(latent)
            
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)

            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)
        
        # [B, hidden_dim] × T → [B, T, hidden_dim]   
        memories = torch.stack(memories, dim=1)
        latents = torch.stack(latents, dim=1)
        
        prior_means = torch.stack(prior_means, dim=1)
        prior_stds = torch.stack(prior_stds, dim=1)
        
        posterior_means = torch.stack(posterior_means, dim=1)
        posterior_stds = torch.stack(posterior_stds, dim=1)
        
        return memories, latents, prior_means, prior_stds, posterior_means, posterior_stds
                
    def imagine(self, initial_memory, initial_latent, actions):
        memory = initial_memory
        latent = initial_latent
        
        memories = []
        latents = []

        prior_means = []
        prior_stds = []
        
        T = actions.shape[1]
        for t in range(T):
            action = actions[:, t]
            memory = self.gru(latent, action, memory)
            prior_mean, prior_std = self.prior(memory)
            latent = self.sample(prior_mean, prior_std)
            
            memories.append(memory)
            latents.append(latent)
            
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            
        memories = torch.stack(memories, dim=1)
        latents = torch.stack(latents, dim=1)
        
        prior_means = torch.stack(prior_means, dim=1)
        prior_stds = torch.stack(prior_stds, dim=1)
        
        return memories, latents, prior_means, prior_stds
            

class Decoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)        
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        recon = self.fc3(x)
        return recon
        
        
class RewardModel(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        reward = self.fc3(x)
        return reward
          

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.encoder = Encoder(state_dim)
        self.rssm = RSSM(action_dim)
        self.decoder = Decoder(state_dim)
        self.reward_model = RewardModel()
        
    def loss(self, states, actions, rewards, initial_memory, initial_latent, alpha=0.8):
        embeds = self.encoder(states)
        memories, latents, prior_means, prior_stds, posterior_means, posterior_stds = self.rssm.observe(embeds, actions, initial_memory, initial_latent)
        
        recons = self.decoder(memories, latents)
        pred_rewards = self.reward_model(memories, latents)
        
        recon_loss = F.mse_loss(states, recons)
        reward_loss = F.mse_loss(rewards, pred_rewards)
        
        prior_dist = torch.distributions.Normal(prior_means, prior_stds)
        posterior_dist = torch.distributions.Normal(posterior_means, posterior_stds)
        
        prior_loss = torch.distributions.kl_divergence(
            torch.distributions.Normal(
                posterior_means.detach(),
                posterior_stds.detach()
            ),
            prior_dist).mean()
            
        posterior_loss = torch.distributions.kl_divergence(
            posterior_dist,
            torch.distributions.Normal(
                prior_means.detach(),
                prior_stds.detach()
            )).mean()
    
    
        dist_loss = alpha * prior_loss + (1 - alpha) * posterior_loss        
        total_loss = recon_loss + reward_loss + dist_loss
                
        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "reward_loss": reward_loss,
            "distribution_loss": dist_loss,
        }
        

class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim=256, latent_dim=32):
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
                
        return mean, std
        
    def sample(self, memory, latent):
        mean, std = self.forward(memory, latent)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        return action


class Critic(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        value = self.out(x)
        return value

        
class Agent:
    def __init__(self, state_dim, action_dim):
        self.world_model = WorldModel(state_dim, action_dim)
        self.actor = Actor(action_dim)
        self.critic = Critic()
        
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        
    def update_world_model(self, states, actions, rewards, initial_memory, initial_latent):
        losses = self.world_model.loss(states, actions, rewards, initial_memory, initial_latent)        
        self.world_model_optimizer.zero_grad()
        losses["total_loss"].backward()
        self.world_model_optimizer.step()
                
        return losses
        
    def imagine_rollout(self, initial_memory, initial_latent, horizon=15):
        memory = initial_memory
        latent = initial_latent
        
        memories = []
        latents = []
        actions = []
        rewards = []
        values = []
        
        for t in range(horizon):
            mean, std = self.actor(memory, latent)
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            
            memory = self.world_model.rssm.gru(latent, action, memory)
            prior_mean, prior_std = self.world_model.rssm.prior(memory)
            
            latent = self.world_model.rssm.sample(prior_mean, prior_std)            
            reward = self.world_model.reward_model(memory, latent)            
            value = self.critic(memory,latent)
            
            memories.append(memory)
            latents.append(latent)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            
        memories = torch.stack(memories, dim=1)
        latents = torch.stack(latents, dim=1)
        actions = torch.stack(actions, dim=1)
        rewards = torch.stack(rewards, dim=1)
        values = torch.stack(values, dim=1)
        
        return memories, latents, actions, rewards, values
        
    def compute_return(self, rewards, values, gamma=0.99, lambda_=0.95):
        B, H, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        next_return = values[:, -1, :]
        
        for t in reversed(range(H)):
            if t == H-1:
                next_value = values[:, -1, :]
            else:
                next_value = values[:, t+1, :]

            next_return = rewards[:, t, :] + gamma * ((1 - lambda_) * next_value + lambda_ * next_return)            
            returns[:, t, :] = next_return
            
        return returns       
        
    def update_critic(self, initial_memory, initial_latent, horizon=15):
        memories, latents, actions, rewards, values = self.imagine_rollout(initial_memory, initial_latent, horizon)
        returns = self.compute_return(rewards, values)
        
        critic_loss = F.mse_loss(values, returns.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss
        
    def update_actor(self, initial_memory, initial_latent, horizon=15):
        for param in self.critic.parameters():
            param.requires_grad = False
            
        memories, latents, actions, rewards, values = self.imagine_rollout(initial_memory, initial_latent, horizon)
        returns = self.compute_return(rewards, values)
        
        actor_loss = -returns.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param in self.critic.parameters():
            param.requires_grad = True
            
        return actor_loss


class ReplayBuffer:
    def __init__(self, max_episodes=1000, batch_size=256, sequence_length=50):
        self.max_episodes = max_episodes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.episodes = []

    def add_episode(self, states, actions, rewards):
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)

        self.episodes.append({
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
        })

    def sample(self):
        states = []
        actions = []
        rewards = []

        while len(states) < self.batch_size:
            episode = np.random.choice(self.episodes)

            if len(episode["states"]) < self.sequence_length:
                continue

            start = np.random.randint(
                0,
                len(episode["states"]) - self.sequence_length + 1
            )

            end = start + self.sequence_length

            states.append(episode["states"][start:end])
            actions.append(episode["actions"][start:end])
            rewards.append(episode["rewards"][start:end])

        return {
            "states": torch.tensor(np.array(states)),
            "actions": torch.tensor(np.array(actions)),
            "rewards": torch.tensor(np.array(rewards)),
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
    buffer = ReplayBuffer()    
    writer = SummaryWriter(log_dir=BASE_DIR)
    
    episode_states = {}
    episode_actions = {}
    episode_rewards = {}
    memories = {}
    latents = {}
    
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)   
        agent_ids = decision_steps.agent_id
        
        for i, agent_id in enumerate(agent_ids):
            if agent_id not in memories:
                memories[agent_id] = torch.zeros(256)
                latents[agent_id] = torch.zeros(32)
                episode_states[agent_id] = []
                episode_actions[agent_id] = []
                episode_rewards[agent_id] = []
        
        if len(agent_ids) > 0:
            states_tensor = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)
            
            actions = []
            with torch.no_grad():
                for i, agent_id in enumerate(agent_ids):
                    action = agent.actor.sample(
                        memories[agent_id].unsqueeze(0),
                        latents[agent_id].unsqueeze(0)
                    )
                    actions.append(action.squeeze(0))
            actions = torch.stack(actions)
            actions = actions.cpu().numpy().astype(np.float32)
            env.set_actions(behavior_name, ActionTuple(continuous=actions))
            
        env.step()
        next_decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        for i, agent_id in enumerate(agent_ids):
            if agent_id in next_decision_steps:
                reward = next_decision_steps[agent_id].reward
                done = False
                next_obs = next_decision_steps[agent_id].obs[0]
            elif agent_id in terminal_steps:
                reward = terminal_steps[agent_id].reward
                done = True
                next_obs = terminal_steps[agent_id].obs[0]
            else:
                continue
            
            episode_states[agent_id].append(states_tensor[i].numpy())    
            episode_actions[agent_id].append(actions[i])
            episode_rewards[agent_id].append(reward)
            
            with torch.no_grad():
                old_memory = memories[agent_id].unsqueeze(0)
                old_latent = latents[agent_id].unsqueeze(0)
                action = torch.from_numpy(actions[i]).float().unsqueeze(0)
                
                new_memory = agent.world_model.rssm.gru(old_latent, action, old_memory)
                next_state = torch.from_numpy(next_obs).float().unsqueeze(0)
                next_embed = agent.world_model.encoder(next_state)
                
                posterior_mean, posterior_std = agent.world_model.rssm.posterior(new_memory, next_embed)
                next_latent = agent.world_model.rssm.sample(posterior_mean, posterior_std)
                
                memories[agent_id] = new_memory.squeeze(0)
                latents[agent_id] = next_latent.squeeze(0)
            
            if done:
                buffer.add_episode(
                    episode_states[agent_id],
                    episode_actions[agent_id],
                    episode_rewards[agent_id]
                )
                
                del episode_states[agent_id]
                del episode_actions[agent_id]
                del episode_rewards[agent_id]
                del memories[agent_id]
                del latents[agent_id]
                 

        
        
        
        
                    

        
        
        
        
