import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler


class ReplayBuffer():
    def __init__(self,
                 buffer_size,
                 device,
                 coreset_size = 30000,
                 exploration_mode = True,
                 ) -> None:
        self.buffer_size = buffer_size
        self.device = device
        self.coreset_size = coreset_size
        self.real_size = 0
        self.iter = 0
        self.exploration_mode = exploration_mode
        self.prioritization = False
        
        # Initialize the buffer
        self.buffer = []
        self.true_rewards = []
        self.max_true_reward = -np.inf
        
    #----------------- For Adding -----------------#
    def add(self, transition, rewards):
        self.iter += 1
        num_input_transitions = transition.shape[0]
        if self.real_size + num_input_transitions >= self.buffer_size:
            self.truncate(num_input_transitions)

        self.max_true_reward = max(self.max_true_reward, rewards.max().item())
        transition, rewards = transition.cpu().numpy(), rewards.cpu().numpy()

        for i in range(num_input_transitions):
            self.buffer.append(transition[i])
            self.true_rewards.append(max(rewards[i], -100))
            self.real_size += 1
            
    #----------------- For Sampling -----------------#
    def sample(self, batch_size):
        if self.prioritization == True:
            prioritization = self.softmax(self.true_rewards)
        else:
            prioritization = np.ones_like(self.true_rewards)
            prioritization /= prioritization.sum()
        indices = np.random.choice(self.real_size, batch_size, p=prioritization, replace=False)

        # Get samples
        samples = np.array(self.buffer)[indices]
        rewards = np.array(self.true_rewards)[indices]
        if self.exploration_mode == True:
            rewards = np.full_like(rewards, self.max_true_reward)
        
        samples, rewards = torch.tensor(samples).float().to(self.device), torch.tensor(rewards).float().to(self.device)
        
        return samples, rewards
    #----------------- For Dropping -----------------#
    
    def truncate(self, num_samples, return_samples=False):
        #dropping_indices = self.get_low_reward_indices(num_samples)
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
    
    #----------------- For Core Set -----------------#
    def get_core_set(self):
        self.exploration_mode = False
        high_reward_indices = np.argsort(self.true_rewards)[-self.coreset_size:]
        self.buffer = list(np.array(self.buffer)[high_reward_indices])
        self.true_rewards = list(np.array(self.true_rewards)[high_reward_indices])
        self.real_size = self.coreset_size
    
    #----------------- For Prioritization -----------------#
    def set_prioritization(self):
        self.exploration_mode = False
        self.prioritization = True    
    
    #----------------- Utils -----------------#
        
    def __len__(self):
        return self.real_size
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    #----------------- For Loading -----------------#
    def save_buffer(self, path):
        np.save(path, np.array(self.buffer), allow_pickle=True)
        np.save(path+'_rewards', np.array(self.true_rewards), allow_pickle=True)
        
    def load_buffer(self, path):
        self.buffer = np.load(path, allow_pickle=True)
        self.true_rewards = np.load(path.replace(".npy", "_rewards.npy"), allow_pickle=True)
        self.real_size = len(self.buffer)
        