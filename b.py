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
    def __init__(self, action_dim, hidden_dim=256, num_categoricals=32, num_classes=32, embed_dim=128):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.latent_dim = num_categoricals * num_classes
        
        input_dim = num_categoricals * num_classes + action_dim
        
        self.W_forget_i = nn.Linear(input_dim, hidden_dim)
        self.W_forget_m = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_info_i = nn.Linear(input_dim, hidden_dim)
        self.W_info_m = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_update_i = nn.Linear(input_dim, hidden_dim)
        self.W_update_m = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_prior = nn.Linear(hidden_dim, hidden_dim)
        self.W_prior_logits = nn.Linear(hidden_dim, num_categoricals * num_classes)    
        
        self.W_posterior = nn.Linear(hidden_dim + embed_dim, hidden_dim)   
        self.W_posterior_logits = nn.Linear(hidden_dim, num_categoricals * num_classes)  
        
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
        logits = self.W_prior_logits(x)
        logits = logits.view(-1, self.num_categoricals, self.num_classes)       
        return logits
        
    def posterior(self, memory, embed):
        x = torch.cat([memory, embed], dim=-1)       
        x = self.W_posterior(x)
        x = F.elu(x)        
        logits = self.W_posterior_logits(x)
        logits = logits.view(-1, self.num_categoricals, self.num_classes)          
        return logits
        
    def sample(self, logits):
        probs = F.softmax(logits, dim=-1)
        indices = torch.argmax(probs, dim=-1)
        sample = F.one_hot(indices, self.num_classes).float()
        sample = sample + probs - probs.detach()
        
        sample = sample.reshape(
            *sample.shape[:-2],
            self.num_categoricals * self.num_classes
        )
        return sample
        
    def observe(self, embeds, actions, initial_memory, initial_latent):
        memory = initial_memory
        latent = initial_latent
        
        memories = []
        latents = []

        prior_logits_list = []
        posterior_logits_list = []
        
        seq_len = embeds.shape[1]  # [B, seq_len, embed_dim]       
        for t in range(seq_len):
            embed = embeds[:, t, :]
            action = actions[:, t, :]
            
            memory = self.gru(latent, action, memory)
            prior_logits = self.prior(memory)
            posterior_logits = self.posterior(memory, embed)
            
            latent = self.sample(posterior_logits)
            
            memories.append(memory)
            latents.append(latent)
            
            prior_logits_list.append(prior_logits)
            posterior_logits_list.append(posterior_logits)
        
        # [B, hidden_dim] × T → [B, T, hidden_dim]   
        memories = torch.stack(memories, dim=1)
        latents = torch.stack(latents, dim=1)
        
        prior_logits = torch.stack(prior_logits_list, dim=1)
        posterior_logits = torch.stack(posterior_logits_list, dim=1)
        
        return memories, latents, prior_logits, posterior_logits                 
            

class Decoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, latent_dim=1024):
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
    def __init__(self, hidden_dim=256, latent_dim=1024):
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
        
        
class DiscountModel(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        discount = torch.sigmoid(self.fc3(x))
        return discount
          

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.encoder = Encoder(state_dim)
        self.rssm = RSSM(action_dim)
        self.decoder = Decoder(state_dim)
        self.reward_model = RewardModel()
        self.discount_model = DiscountModel()
        
    def loss(self, states, actions, rewards, discounts, initial_memory, initial_latent, alpha=0.8):
        embeds = self.encoder(states)
        memories, latents, prior_logits, posterior_logits = self.rssm.observe(embeds, actions, initial_memory, initial_latent)
        
        recons = self.decoder(memories, latents)
        pred_rewards = self.reward_model(memories, latents)
        pred_discounts = self.discount_model(memories, latents)
        
        recon_loss = F.mse_loss(states, recons)
        reward_loss = F.mse_loss(rewards.unsqueeze(-1), pred_rewards)
        discount_loss = F.binary_cross_entropy(pred_discounts, discounts.unsqueeze(-1))
        
        prior_loss = torch.distributions.kl_divergence(
            torch.distributions.Categorical(
                logits=posterior_logits.detach()
            ),
            torch.distributions.Categorical(
                logits=prior_logits
            )
        ).sum(dim=-1).mean()
            
        posterior_loss = torch.distributions.kl_divergence(
            torch.distributions.Categorical(
                logits=posterior_logits
            ),
            torch.distributions.Categorical(
                logits=prior_logits.detach()
            )
        ).sum(dim=-1).mean() 
    
        dist_loss = alpha * prior_loss + (1 - alpha) * posterior_loss        
        total_loss = recon_loss + reward_loss + discount_loss + dist_loss
                
        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "reward_loss": reward_loss,
            "discount_loss": discount_loss,
            "distribution_loss": dist_loss,            
            "memories": memories,
            "latents": latents,
        }
        

class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim=256, latent_dim=1024):
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
        action = torch.tanh(action)
        return action
        
    def deterministic(self, memory, latent):
        mean, _ = self.forward(memory, latent)
        mean = torch.tanh(mean)
        return mean


class Critic(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=1024):
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
        
    def update_world_model(self, states, actions, rewards, discounts, initial_memory, initial_latent):           
        losses = self.world_model.loss(states, actions, rewards, discounts, initial_memory, initial_latent)        
        self.world_model_optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 10.0)
        self.world_model_optimizer.step()
                        
        return losses
        
    def imagine(self, initial_memory, initial_latent, horizon=15):
        memory = initial_memory
        latent = initial_latent
        
        memories = []
        latents = []
        actions = []
        rewards = []
        values = []
        discounts = []
        
        for t in range(horizon):
            action = self.actor.sample(memory, latent)
            
            memory = self.world_model.rssm.gru(latent, action, memory)
            prior_logits = self.world_model.rssm.prior(memory)
            
            latent = self.world_model.rssm.sample(prior_logits)            
            reward = self.world_model.reward_model(memory, latent)
            discount = self.world_model.discount_model(memory, latent)            
            value = self.critic(memory, latent)
            
            memories.append(memory)
            latents.append(latent)
            actions.append(action)
            rewards.append(reward)
            discounts.apped(discount)
            values.append(value)
            
        memories = torch.stack(memories, dim=1)
        latents = torch.stack(latents, dim=1)
        actions = torch.stack(actions, dim=1)
        rewards = torch.stack(rewards, dim=1)
        discounts = torch.stack(discounts, dim=1)
        
        action = self.actor.sample(memory, latent)
        memory = self.world_model.rssm.gru(latent, action, memory)
        prior_logits = self.world_model.rssm.prior(memory)
        latent = self.world_model.rssm.sample(prior_logits) 
        value = self.critic(memory, latent)
        values.append(value)
        
        values = torch.stack(values, dim=1)
        
        return memories, latents, actions, rewards, discounts, values
        
    def compute_return(self, rewards, values, discounts, lambda_=0.95):
        B, H, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        next_return = values[:, -1, :]
        
        for t in reversed(range(H)):
            if t == H-1:
                next_value = values[:, -1, :]
            else:
                next_value = values[:, t+1, :]

            next_return = rewards[:, t, :] + discounts[:, t, :] * ((1 - lambda_) * next_value + lambda_ * next_return)            
            returns[:, t, :] = next_return
            
        return returns       
        
    def update_critic(self, initial_memory, initial_latent, horizon=15):
        for param in self.actor.parameters():
            param.requires_grad = False

        for param in self.world_model.parameters():
            param.requires_grad = False
            
        memories, latents, actions, rewards, discounts, values = self.imagine(initial_memory, initial_latent, horizon)
        returns = self.compute_return(rewards, values, discounts)
        
        critic_loss = F.mse_loss(values[:, :-1, :], returns.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        for param in self.actor.parameters():
            param.requires_grad = True

        for param in self.world_model.parameters():
            param.requires_grad = True
        
        return critic_loss
        
    def update_actor(self, initial_memory, initial_latent, horizon=15):
        for param in self.critic.parameters():
            param.requires_grad = False
            
        for param in self.world_model.parameters():
            param.requires_grad = False
            
        memories, latents, actions, rewards, discounts, values = self.imagine(initial_memory, initial_latent, horizon)
        returns = self.compute_return(rewards, values, discounts)
        
        actor_loss = -returns.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        for param in self.critic.parameters():
            param.requires_grad = True
            
        for param in self.world_model.parameters():
            param.requires_grad = True
            
        return actor_loss


class ReplayBuffer:
    def __init__(self, max_episodes=1000, batch_size=50, sequence_length=50):
        self.max_episodes = max_episodes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.episodes = []
        
    def update_sequence_length(self):
        max_length = max(len(ep["states"]) for ep in self.episodes)
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
            self.sequence_length = 5      

    def add_episode(self, states, actions, rewards):
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)
            
        discounts = np.full(len(rewards), 0.999, dtype=np.float32)
        discounts[-1] = 0.0

        self.episodes.append({
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "discounts": discounts,
        })

    def sample(self):
        states = []
        actions = []
        rewards = []
        discounts = []

        while len(states) < self.batch_size:
            episode = np.random.choice(self.episodes)
            episode_length = len(episode["states"])
            if episode_length < self.sequence_length:
                continue

            start = np.random.randint(0, episode_length)
            start = min(
                start,
                episode_length - self.sequence_length
            )

            end = start + self.sequence_length

            states.append(episode["states"][start:end])
            actions.append(episode["actions"][start:end])
            rewards.append(episode["rewards"][start:end])
            discounts.append(episode["discounts"][start:end])

        return {
            "states": torch.tensor(np.array(states), dtype=torch.float32),
            "actions": torch.tensor(np.array(actions), dtype=torch.float32),
            "rewards": torch.tensor(np.array(rewards), dtype=torch.float32),
            "discounts": torch.tensor(np.array(discounts), dtype=torch.float32),
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
    
    #model = torch.load(f"{BASE_DIR}/period_model.pth")
    #agent.world_model.encoder.load_state_dict(model["encoder"])
    #agent.world_model.rssm.load_state_dict(model["rssm"])
    #agent.actor.load_state_dict(model["actor"])
    
    memory_dim, latent_dim = 256, 1024
    train_max_step, test_max_step = 1000, 1000
    min_buffer_size = 50
    updates_per_episode = 10
    test_interval = 10
    
    episode_states = {}
    episode_actions = {}
    episode_rewards = {}
    episode_steps = {}
    memories = {}
    latents = {}
    
    update_iteration = 0
    save_idx = 0
    best_test_reward = -float('inf')
    
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)   
        agent_ids = decision_steps.agent_id
        
        for i, agent_id in enumerate(agent_ids):
            if agent_id not in memories:
                memories[agent_id] = torch.zeros(memory_dim)
                latents[agent_id] = torch.zeros(latent_dim)
                episode_states[agent_id] = []
                episode_actions[agent_id] = []
                episode_rewards[agent_id] = []
                episode_steps[agent_id] = 0
        
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
            actions_np = actions.cpu().numpy().astype(np.float32)
            env.set_actions(behavior_name, ActionTuple(continuous=actions_np))
            
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
                
            episode_steps[agent_id] += 1
            if episode_steps[agent_id] >= train_max_step:
                done = True
            
            episode_states[agent_id].append(next_obs)  
            episode_actions[agent_id].append(actions[i].detach().cpu().numpy())
            episode_rewards[agent_id].append(reward)
            
            with torch.no_grad():
                old_memory = memories[agent_id].unsqueeze(0)
                old_latent = latents[agent_id].unsqueeze(0)
                action = actions[i].float().unsqueeze(0)
                
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
                buffer.update_sequence_length()                
                
                print("episode_length:", len(episode_rewards[agent_id]))
                
                del episode_states[agent_id]
                del episode_actions[agent_id]
                del episode_rewards[agent_id]
                del memories[agent_id]
                del latents[agent_id]
                del episode_steps[agent_id]
        
                if len(buffer.episodes) >= min_buffer_size:
                    for _ in range(updates_per_episode):
                        batch = buffer.sample()
            
                        batch_states = batch["states"]
                        batch_actions = batch["actions"]
                        batch_rewards = batch["rewards"]
                        batch_discounts = batch["discounts"]  
            
                        B = batch_states.shape[0]        
                        initial_memory = torch.zeros(B, memory_dim)
                        initial_latent = torch.zeros(B, latent_dim)       

                        losses = agent.update_world_model(batch_states, batch_actions, batch_rewards, batch_discounts, initial_memory, initial_latent)
            
                        writer.add_scalar("Train/WorldModel_total_loss", losses["total_loss"].item(), update_iteration)
                        writer.add_scalar("Train/WorldModel_reconstruction_loss", losses["reconstruction_loss"].item(), update_iteration)
                        writer.add_scalar("Train/WorldModel_reward_loss", losses["reward_loss"].item(), update_iteration)
                        writer.add_scalar("Train/WorldModel_discount_loss", losses["discount_loss"].item(), update_iteration)
                        writer.add_scalar("Train/WorldModel_distribution_loss", losses["distribution_loss"].item(), update_iteration)
            
                        imagination_memory = losses["memories"][:, -1, :]
                        imagination_latent = losses["latents"][:, -1, :]        

                        critic_loss = agent.update_critic(imagination_memory.detach(), imagination_latent.detach())
                        writer.add_scalar("Train/Critic_loss", critic_loss.item(), update_iteration)
            
                        actor_loss = agent.update_actor(imagination_memory.detach(), imagination_latent.detach())
                        writer.add_scalar("Train/Actor_loss", actor_loss.item(), update_iteration)      
                    
                    update_iteration += 1
                                      
                    if update_iteration % test_interval == 0:
                        print(f"[Test] update_iteration {update_iteration}")
                        test_env.reset()
                        t_decision_steps, _ = test_env.get_steps(t_behavior_name)
                
                        n_test_agents = len(t_decision_steps.agent_id)
                        test_rewards = np.zeros(n_test_agents)
                        test_episode_dones = np.zeros(n_test_agents, dtype=bool)
                        test_id_to_index = {agent_id: i for i, agent_id in enumerate(t_decision_steps.agent_id)}                 
                
                        test_memories = {}
                        test_latents = {}
                        test_total_reward = 0.0

                        for agent_id in t_decision_steps.agent_id:
                            test_memories[agent_id] = torch.zeros(1, memory_dim)
                            test_latents[agent_id] = torch.zeros(1, latent_dim)
                
                        test_max_step_count = 0    
                        while not np.all(test_episode_dones) and test_max_step_count < test_max_step:
                            t_decision_steps, _ = test_env.get_steps(t_behavior_name)
                            t_agent_ids = t_decision_steps.agent_id
                            if len(t_agent_ids) > 0:
                                t_states_tensor = torch.from_numpy(t_decision_steps.obs[0]).to(torch.float32)   
                                t_actions = []
                                with torch.no_grad():
                                    for i, agent_id in enumerate(t_agent_ids):
                                        memory = test_memories[agent_id]
                                        latent = test_latents[agent_id]
                                        t_action = agent.actor.deterministic(memory, latent)
                                        t_actions.append(t_action.squeeze(0))
                                
                                t_actions = torch.stack(t_actions)                     
                                t_actions_np = t_actions.cpu().numpy().astype(np.float32)
                                test_env.set_actions(t_behavior_name, ActionTuple(continuous=t_actions_np))
                        
                            test_env.step()
                            test_max_step_count += 1
                            t_next_decision_steps, t_terminal_steps = test_env.get_steps(t_behavior_name)
                    
                            for i, agent_id in enumerate(t_agent_ids):
                                if agent_id in t_next_decision_steps:
                                    reward = t_next_decision_steps[agent_id].reward
                                    done = False
                                    next_obs = t_next_decision_steps[agent_id].obs[0]
                                elif agent_id in t_terminal_steps:
                                    reward = t_terminal_steps[agent_id].reward
                                    done = True
                                    next_obs = t_terminal_steps[agent_id].obs[0]
                            
                                    idx = test_id_to_index[agent_id]
                                    test_episode_dones[idx] = True
                                else:
                                    continue
                            
                                idx = test_id_to_index[agent_id]
                                test_rewards[idx] += reward                           
                        
                                with torch.no_grad():
                                    old_memory = test_memories[agent_id]
                                    old_latent = test_latents[agent_id]
                        
                                    t_action = t_actions[i].unsqueeze(0)
                                    new_memory = agent.world_model.rssm.gru(old_latent, t_action, old_memory)
                                    next_state = torch.from_numpy(next_obs).float().unsqueeze(0)
                                    next_embed = agent.world_model.encoder(next_state)
                            
                                    posterior_mean, posterior_std = agent.world_model.rssm.posterior(new_memory, next_embed)                            
                                    next_latent = posterior_mean
                           
                                    test_memories[agent_id] = new_memory
                                    test_latents[agent_id] = next_latent

                        test_average_reward = np.mean(test_rewards)  
                        writer.add_scalar("Test/Average_Reward", test_average_reward, update_iteration)
                        print(f"[Test] {test_average_reward:.4f}")
                        torch.save({
                            "world_model": agent.world_model.state_dict(),
                            "actor": agent.actor.state_dict(),
                            "critic": agent.critic.state_dict(),
                            "world_model_optimizer": agent.world_model_optimizer.state_dict(),
                            "actor_optimizer": agent.actor_optimizer.state_dict(),
                            "critic_optimizer": agent.critic_optimizer.state_dict(),
                        }, f"{BASE_DIR}/checkpoint.pth")
               
                        torch.save({
                            "encoder": agent.world_model.encoder.state_dict(),
                            "rssm": agent.world_model.rssm.state_dict(),
                            "actor": agent.actor.state_dict(),
                        }, f"{BASE_DIR}/period_model.pth")
                        
                        if test_average_reward > best_test_reward:
                            best_test_reward = test_average_reward
                            save_idx += 1
                            torch.save({
                                "encoder": agent.world_model.encoder.state_dict(),
                                "rssm": agent.world_model.rssm.state_dict(),
                                "actor": agent.actor.state_dict(),
                            }, f"{BASE_DIR}/#({save_idx})best_{best_test_reward:.4f}.pth")
                            print(f"[Test] Model saved at new best reward {best_test_reward:.4f}")                           
        
