import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.predictor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        self.target = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, transition):
        target_feature = self.target(transition)
        predict_feature = self.predictor(transition)

        return predict_feature, target_feature
    
    def predict_error(self, transition):
        predict_feature, target_feature = self.forward(transition)
        return (predict_feature - target_feature).pow(2).sum(1) # if transition (100, 32) -> (100, 1)
    
class ReplayBuffer():
    def __init__(self,
                 buffer_size,
                 device,
                 rnd_input = 2,
                 rnd_output = 1,
                 learning_rate = 1e-6,
                 exploration_mode = True,
                 rewards_type = 'mixed'
                 ) -> None:
        self.buffer_size = buffer_size
        self.device = device
        self.real_size = 0
        self.iter = 0
        self.exploration_mode = exploration_mode
        # Initialize the buffer
        self.buffer = []
        self.true_rewards = []
        self.intrinsic_rewards = []
        # Rewards Type
        self.rewards_type = rewards_type
        if self.rewards_type not in ["true", "intrinsic", "mixed"]:
            raise ValueError("Invalid rewards type. Choose from 'true', 'intrinsic', 'mixed'")
        
        # Initialize RND model for intrinsic reward
        self.rnd_model = RNDModel(rnd_input, rnd_output).to(device)
        self.rnd_optimizer = torch.optim.Adam(self.rnd_model.parameters(), lr=learning_rate)
        
        
    def add(self, transition, rewards):
        self.iter += 1
        num_input_transitions = transition.shape[0]
        if self.real_size + num_input_transitions >= self.buffer_size:
            self.truncate(num_input_transitions)
        
        # Reward Clamping to [-100, ]
        rewards = torch.clamp(rewards, min=-100)
        
        # Predict intrinsic reward
        with torch.no_grad():
            intrinsic_reward = self.rnd_model.predict_error(transition)
        transition, rewards, intrinsic_reward = transition.cpu().numpy(), rewards.cpu().numpy(), intrinsic_reward.cpu().numpy()
        
        for i in range(num_input_transitions):
            self.buffer.append(transition[i])
            self.true_rewards.append(rewards[i])
            self.intrinsic_rewards.append(intrinsic_reward[i])
            self.real_size += 1
            
    def sample(self, batch_size):
        indices = self.get_priority(batch_size) # np.array ()
        
        # Get samples
        samples = np.array(self.buffer)[indices]
        true_rewards = np.array(self.true_rewards)[indices]
        intrinsic_rewards = np.array(self.intrinsic_rewards)[indices]
        
        # Train RND model
        self.train_rnd(samples)
        self.update_intrinsic_reward()
        
        rewards = self.get_rewards(true_rewards, intrinsic_rewards)
        # For mixed rewards
        
        samples, rewards = torch.tensor(samples).float().to(self.device), torch.tensor(rewards).float().to(self.device)
        
        return samples, rewards
        
    def get_priority(self, batch_size):
        if self.exploration_mode == True:
            priorities = self.softmax(self.intrinsic_rewards)
        else:
            priorities = self.softmax(self.true_rewards)
        indices = np.random.choice(self.real_size, batch_size, p=priorities, replace=False)
        return indices
        
    def get_rewards(self, true_rewards, intrinsic_rewards):
        if self.rewards_type == "true":
            return true_rewards
        elif self.rewards_type == "intrinsic":
            return intrinsic_rewards
        else:
            return true_rewards + intrinsic_rewards
        
    def truncate(self, num_samples, return_samples=False):
        dropping_indices = self.get_low_reward_indices(num_samples)
        if return_samples:
            return torch.tensor(np.array(self.buffer)[dropping_indices]).float().to(self.device)
        # Drop the lowest intrinsic reward samples
        self.buffer = np.delete(self.buffer, dropping_indices, axis=0)
        self.true_rewards = np.delete(self.true_rewards, dropping_indices)
        self.intrinsic_rewards = np.delete(self.intrinsic_rewards, dropping_indices)
        self.real_size -= num_samples    
    
    #----------------- For testing -----------------#
    def get_low_reward_indices(self, num_samples):
        return np.argsort(self.true_rewards)[:num_samples]
    
    def get_low_mixed_reward_indices(self, num_samples):
        return np.argsort(self.true_rewards + self.intrinsic_rewards)[:num_samples]
    
    def get_low_intrinsic_reward_indices(self, num_samples):
        return np.argsort(self.intrinsic_rewards)[:num_samples]
    
    def get_high_reward_indices(self, num_samples):
        return np.argsort(self.true_rewards)[-num_samples:]
    
    def get_high_mixed_reward_indices(self, num_samples):
        return np.argsort(self.true_rewards + self.intrinsic_rewards)[-num_samples:]
    
    def get_high_intrinsic_reward_indices(self, num_samples):
        return np.argsort(self.intrinsic_rewards)[-num_samples:]        
    
    #----------------- Utils -----------------#
        
    def __len__(self):
        return len(self.buffer)
        
    #----------------- RND and Prioritization-----------------#
    def train_rnd(self, transition) -> float:
        transition = torch.tensor(transition).float().to(self.device)
        self.rnd_optimizer.zero_grad()
        intrinsic_reward = self.rnd_model.predict_error(transition)
        loss = intrinsic_reward.mean()
        loss.backward()
        self.rnd_optimizer.step()
        
        return loss.item()
    
    def update_intrinsic_reward(self):
        temp = torch.tensor(self.buffer[:self.real_size]).float().to(self.device)
        intrinsic_reward = self.rnd_model.predict_error(temp).cpu().detach().numpy()
        self.intrinsic_rewards[:self.real_size] = intrinsic_reward
        
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()