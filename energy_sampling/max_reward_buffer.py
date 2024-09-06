import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler


class ReplayBuffer():
    def __init__(self,
                 buffer_size,
                 device,
                 exploration_mode = True,
                 negligible_reward = -100,
                 ) -> None:
        self.buffer_size = buffer_size
        self.device = device
        self.real_size = 0
        self.iter = 0
        self.exploration_mode = exploration_mode
        self.prioritization = False
        self.min_cut_value = negligible_reward
        
        self.min_value_truncate = True
        
        self.higher_reward = 100
        self.lower_reward = -100
        
        self.min_reward = negligible_reward
        
        # Initialize the buffer
        self.buffer = []
        self.true_rewards = []
        
                
    #----------------- For Adding -----------------#
    def add(self, transition, rewards):
        self.iter += 1
        if self.min_value_truncate:
            if self.iter != 1:
                self.min_reward = min(self.true_rewards)
            allowing_indices = torch.where(rewards > self.min_reward)[0]
            transition, rewards = transition[allowing_indices], rewards[allowing_indices]
        len_adding = len(rewards)
        if self.real_size + len_adding > self.buffer_size:
            self.truncate(len_adding)

        transition, rewards = transition.cpu().numpy(), rewards.cpu().numpy()

        for i in range(len_adding):
            self.buffer.append(transition[i])
            self.true_rewards.append(rewards[i])
            self.real_size += 1
            
    #----------------- For Sampling -----------------#
    def sample(self, batch_size):
        batch_size = min(batch_size, self.real_size)
        if self.prioritization == True:
            self.sampler = WeightedRandomSampler(self.weights, batch_size, replacement=False)
            indices = list(self.sampler)
        else:
            prioritization = np.ones_like(self.true_rewards)
            prioritization /= prioritization.sum()
            indices = np.random.choice(self.real_size, batch_size, p=prioritization, replace=False)

        # Get samples
        samples = np.array(self.buffer)[indices]
        rewards = np.array(self.true_rewards)[indices]
        
        if self.exploration_mode == True:
            low_reward_indices = np.where(rewards < self.min_cut_value)[0]
            high_reward_indices = np.where(rewards >= self.min_cut_value)[0]
            rewards[low_reward_indices] = self.lower_reward
            rewards[high_reward_indices] = self.higher_reward
            
        samples, rewards = torch.tensor(samples).float().to(self.device), torch.tensor(rewards).float().to(self.device)
        
        print("Minimum Reward in buffer: ", self.min_reward)
        return samples, rewards
    #----------------- For Dropping -----------------#
    
    def truncate(self, num_samples, return_samples=False):
        num_samples = min(num_samples, self.real_size)
        print("buffer full")
        
        if self.min_value_truncate:
            dropping_indices = self.get_low_reward_indices(num_samples)
        else:
            dropping_indices = np.random.choice(self.real_size, num_samples, replace=False)
        if return_samples:
            return torch.tensor(np.array(self.buffer)[dropping_indices]).float().to(self.device)
        # Drop the lowest intrinsic reward samples
        self.buffer = list(np.delete(self.buffer, dropping_indices, axis=0))
        self.true_rewards = list(np.delete(self.true_rewards, dropping_indices))
        self.real_size -= num_samples    
    
    #----------------- Dropping Utils -----------------#
    def get_low_reward_indices(self, num_samples):
        return np.argsort(self.true_rewards)[:num_samples] 
    
    #----------------- For Prioritization -----------------#
    def set_prioritization(self):
        self.exploration_mode = False
        self.prioritization = True      
        ranks = np.argsort(np.argsort(-1 * np.array(self.true_rewards)))
        self.weights = 1.0 / (1e-2 * len(np.array(self.true_rewards)) + ranks)
            
    #----------------- Utils -----------------#
        
    def __len__(self):
        return self.real_size
    
    #----------------- For Loading -----------------#
    def save_buffer(self, path):
        np.save(path, np.array(self.buffer), allow_pickle=True)
        np.save(path+'_rewards', np.array(self.true_rewards), allow_pickle=True)
        
    #----------------- For Saving -----------------#    
    def load_buffer(self, path):
        self.buffer = np.load(path, allow_pickle=True)
        self.true_rewards = np.load(path.replace(".npy", "_rewards.npy"), allow_pickle=True)
        self.real_size = len(self.buffer)
        