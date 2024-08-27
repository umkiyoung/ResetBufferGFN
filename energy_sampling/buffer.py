import numpy as np
import torch
from collections import deque
from sklearn.neighbors import NearestNeighbors
class ReplayBuffer():
    def __init__(self, max_size, device, knn_k=5, n_epochs=10000, alpha = 1):
        self.max_size = max_size
        self.buffer = []
        self.rewards = []
        self.priorities = []
        self.device = device
        self.knn_k = knn_k  # KNN에서 사용할 이웃의 수
        self.reward_cutline = None
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.iter = 0

    def add(self, transition, rewards):
        self.iter += 1
        transition, rewards = transition.cpu().numpy(), rewards.cpu().numpy()
        if len(self.buffer) + transition.shape[0] >= self.max_size:
            self.remove_high_density_sample(len(self.buffer)//5)
        
        for i in range(transition.shape[0]):
            if self.reward_cutline is not None and rewards[i] < self.reward_cutline:
                continue
            self.buffer.append(transition[i])
            self.rewards.append(max(rewards[i], -100))
            self.priorities.append(-100)

    def sample(self, batch_size):
        priorities = self.softmax(self.priorities)
        
        # Replace=False로 중복되지 않게 샘플링
        try:
            indices = np.random.choice(len(self.buffer), batch_size, p=priorities, replace=False)
        except:
            indices = np.random.choice(len(self.buffer), batch_size, p = self.softmax(self.rewards), replace=False)
        
        samples = [self.buffer[idx] for idx in indices]
        rewards  = [self.priorities[idx] for idx in indices]
        
        # Tensor로 변환 후 장치로 이동
        samples, rewards = torch.tensor(np.array(samples)).float().to(self.device), torch.tensor(np.array(rewards)).float().to(self.device)
        
        return samples, rewards
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def remove_high_density_sample(self, num_samples, return_samples=False):
        # Buffer에 있는 샘플들에 대해 KNN을 사용하여 밀도를 계산
        buffer_array = np.array(self.buffer)
        if len(buffer_array) < self.knn_k + 1:
            return  # 샘플이 너무 적으면 제거하지 않음

        nbrs = NearestNeighbors(n_neighbors=self.knn_k + 1).fit(buffer_array)
        distances, _ = nbrs.kneighbors(buffer_array)
        
        # 각 샘플의 밀도를 계산 (거리의 역수를 밀도로 간주)
        densities = 1.0 / np.sum(distances[:, 1:], axis=1)
        
        normalized_densities = self.softmax(densities) 
        normalized_rewards = self.softmax(self.rewards)
        
        self.priorities = list(-np.log(normalized_densities + 1e-5)* self.alpha * (self.n_epochs - self.iter / self.n_epochs)  + np.log(normalized_rewards + 1e-5))
        criteria = -np.array(self.rewards)
        high_density_low_reward_indexes = np.argsort(criteria)[-num_samples:] 
            
        if return_samples == True:
            samples = [self.buffer[i] for i in high_density_low_reward_indexes]
            return torch.tensor(np.array(samples)).float().to(self.device)
        else:
            self.buffer = [v for i, v in enumerate(self.buffer) if i not in high_density_low_reward_indexes]
            self.rewards = [v for i, v in enumerate(self.rewards) if i not in high_density_low_reward_indexes]
            self.priorities = [v for i, v in enumerate(self.priorities) if i not in high_density_low_reward_indexes]

    def __len__(self):
        return len(self.buffer)