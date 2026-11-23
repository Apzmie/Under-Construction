import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 

                       
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.silu(self.fc1(state))
        x = F.silu(self.fc2(x))       
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
        action = torch.tanh(mean)
        return action
        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, num_quantiles)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        quantiles = self.q(x)
        return quantiles
        
        
class Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4, num_critics=3, num_quantiles=15, top_quantiles_to_drop=6):
        self.actor = Actor(state_dim, action_dim)        
        self.critics = nn.ModuleList([
            Critic(state_dim, action_dim, num_quantiles) for _ in range(num_critics)
        ])        
        self.target_critics = nn.ModuleList([
            Critic(state_dim, action_dim, num_quantiles) for _ in range(num_critics)
        ])
        
        for critic, target_critic in zip(self.critics, self.target_critics):
            target_critic.load_state_dict(critic.state_dict())
            for p in target_critic.parameters():
                p.requires_grad = False
        
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizers = [
            torch.optim.AdamW(critic.parameters(), lr=lr) for critic in self.critics
        ]

        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=lr)
        
        self.target_entropy = -action_dim
        self.num_critics = num_critics
        self.num_quantiles = num_quantiles
        self.top_quantiles_to_drop = top_quantiles_to_drop
        self.min_drop = num_critics
        self.max_drop = num_quantiles       
    
    def update_target_critic(self, tau=0.005):
        with torch.no_grad():
            for critic, target_critic in zip(self.critics, self.target_critics):
                for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                    target_param.copy_(
                        tau * param + (1 - tau) * target_param
                    )

    def update(self, batch, gamma=0.99):
        state = torch.FloatTensor(batch['state'])
        action = torch.FloatTensor(batch['action'])
        reward = torch.FloatTensor(batch['reward'])
        next_state = torch.FloatTensor(batch['next_state'])
        done = torch.FloatTensor(batch['done'])
        
        #==========================================
        
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            
            next_quantiles_list = [target_critic(next_state, next_action) for target_critic in self.target_critics]
            next_quantiles = torch.stack(next_quantiles_list, dim=1)  # (batch_size, num_critics, num_quantiles)
            batch_size = state.shape[0]            
            next_quantiles = next_quantiles.reshape(batch_size, -1)  # (batch_size, num_critics * num_quantiles)
            
            sorted_next_quantiles, _ = torch.sort(next_quantiles, dim=-1)
            num_atoms_to_keep = (self.num_critics * self.num_quantiles) - self.top_quantiles_to_drop
            truncated_next_quantiles = sorted_next_quantiles[:, :num_atoms_to_keep]  # (batch_size, num_atoms_to_keep)
            
            alpha = self.log_alpha.exp()            
            target_quantiles = reward + gamma * (1 - done) * (truncated_next_quantiles - alpha * next_log_prob)

        current_quantiles_list = [critic(state, action) for critic in self.critics]
        critic_losses = []
        total_one_count = 0
        total_zero_count = 0        
        for n, critic_optimizer in enumerate(self.critic_optimizers):
            pred_quantiles = current_quantiles_list[n]  # (batch_size, num_quantiles)            
            error = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)  # (batch_size, 1, num_atoms_to_keep) - (batch_size, num_quantiles, 1) = (batch_size, num_quantiles, num_atoms_to_keep)  
            smooth_l1_loss = torch.where(torch.abs(error) <= 1.0, 0.5 * error ** 2, 1.0 * (torch.abs(error) - 0.5))  # if 0.5 -> 0.125, if 1.5 -> 1
            
            quantile_levels = torch.arange(0.5, self.num_quantiles, dtype=torch.float32).view(1, -1, 1) / self.num_quantiles  # (1, num_quantiles, 1)  
            quantile_weight = torch.abs(quantile_levels - (error < 0).float())  # if (error < 0) True = 1 -> overestimation, if False = 0 -> underestimation                         
            elementwise_loss = smooth_l1_loss * quantile_weight
            
            critic_loss = elementwise_loss.sum(dim=-1).mean()  # scalar                            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
                        
            critic_losses.append(critic_loss.item())
            
            with torch.no_grad():
                zero_and_one = error < 0
                one_count = zero_and_one.sum().item()
                zero_count = zero_and_one.numel() - one_count
                total_one_count += one_count
                total_zero_count += zero_count
                
        if total_one_count > total_zero_count:
            self.top_quantiles_to_drop = min(self.top_quantiles_to_drop + 1, self.max_drop)
        elif total_zero_count > total_one_count:
            self.top_quantiles_to_drop = max(self.top_quantiles_to_drop - 1, self.min_drop)
        
        #==========================================
        
        for critic in self.critics:
            for p in critic.parameters():
                p.requires_grad = False
        
        action_new, log_prob = self.actor.sample(state)
        
        quantiles_list = [critic(state, action_new) for critic in self.critics]
        quantiles = torch.stack(quantiles_list, dim=1)  # (batch_size, num_critics, num_quantiles)
        quantiles = quantiles.reshape(batch_size, -1)  # (batch_size, num_critics * num_quantiles)
        q_value = quantiles.mean(dim=-1, keepdim=True)  # (batch_size, 1)  
        
        alpha = self.log_alpha.exp().detach()    
        actor_loss = -(q_value - alpha * log_prob).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for critic in self.critics:
            for p in critic.parameters():
                p.requires_grad = True
        
        #==========================================
        
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        #==========================================
        
        self.update_target_critic()
        
        return {
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "alpha": self.log_alpha.exp().item(),
            "target_quantiles": target_quantiles.mean().item(),
            "q_value": q_value.mean().item(),
            "top_quantiles_to_drop": self.top_quantiles_to_drop,
        }
        
    def save_checkpoint(self, path, buffer):
        torch.save({
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),

            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizers": [opt.state_dict() for opt in self.critic_optimizers],
            "alpha_optimizer": self.alpha_optimizer.state_dict(),

            "log_alpha": self.log_alpha.detach().cpu(),

            "replay_buffer": {
                "state": buffer.state,
                "action": buffer.action,
                "reward": buffer.reward,
                "next_state": buffer.next_state,
                "done": buffer.done,
                "ptr": buffer.ptr,
                "size": buffer.size,
            },
        }, path)

    def load_checkpoint(self, path, buffer):
        ckpt = torch.load(path)

        self.actor.load_state_dict(ckpt["actor"])
        self.critics.load_state_dict(ckpt["critics"])
        self.target_critics.load_state_dict(ckpt["target_critics"])

        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        for opt, opt_state in zip(self.critic_optimizers, ckpt["critic_optimizers"]):
            opt.load_state_dict(opt_state)
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])

        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"])

        buffer.state = ckpt["replay_buffer"]["state"]
        buffer.action = ckpt["replay_buffer"]["action"]
        buffer.reward = ckpt["replay_buffer"]["reward"]
        buffer.next_state = ckpt["replay_buffer"]["next_state"]
        buffer.done = ckpt["replay_buffer"]["done"]
        buffer.ptr = ckpt["replay_buffer"]["ptr"]
        buffer.size = ckpt["replay_buffer"]["size"]
        
        
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
