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
        self.min_cut_value = -100
        
        # Initialize the buffer
        self.buffer = []
        self.true_rewards = []
        self.max_true_reward = -100
        
    #----------------- For Adding -----------------#
    def add(self, transition, rewards):
        self.iter += 1
        # -100보다 큰 reward만 저장
        indices = torch.where(rewards > self.min_cut_value)[0]
        transition, rewards = transition[indices], rewards[indices]
        num_input_transitions = len(rewards)
        if len(rewards) == 0:
            return
        if self.real_size + num_input_transitions > self.buffer_size:
            self.truncate(num_input_transitions)

        self.max_true_reward = max(self.max_true_reward, rewards.max().item())
        transition, rewards = transition.cpu().numpy(), rewards.cpu().numpy()

        for i in range(num_input_transitions):
            self.buffer.append(transition[i])
            self.true_rewards.append(rewards[i])
            self.real_size += 1
            
    #----------------- For Sampling -----------------#
    def sample(self, batch_size):
        batch_size = min(batch_size, self.real_size)
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
        num_samples = min(num_samples, self.real_size)
        dropping_indices = self.get_low_reward_indices(num_samples)
        if return_samples == False:
            self.min_cut_value = max(self.min_cut_value, max(np.array(self.true_rewards)[dropping_indices]))
        print(self.min_cut_value)
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
        #high_reward_indices = np.argsort(self.true_rewards)[-self.coreset_size:]
        self.mean_field_histogram(100)
        low_density_indices = np.argsort(self.density_sum)[:self.coreset_size]
        self.buffer = list(np.array(self.buffer)[low_density_indices])
        self.true_rewards = list(np.array(self.true_rewards)[low_density_indices])

        self.real_size = len(self.buffer)
    
    #----------------- For Prioritization -----------------#
    def set_prioritization(self):
        self.exploration_mode = False
        self.prioritization = True    
    
    #----------------- For Histogram -----------------#
    def mean_field_histogram(self, num_bins=100):
        for i in range(np.array(self.buffer).shape[1]):
            dimension_data = self.buffer[:, i]
            hist, bin_edges = np.histogram(dimension_data, bins=num_bins, density=True)
            bin_indices = np.digitize(dimension_data, bin_edges[:-1], right=True)
            rewards_for_dimension = hist[bin_indices -1]
            if i == 0:
                density_sum = rewards_for_dimension
            else:
                density_sum += rewards_for_dimension
        
        self.density_sum = density_sum            
    
    
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
        