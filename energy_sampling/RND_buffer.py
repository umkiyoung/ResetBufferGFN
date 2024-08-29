import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from tqdm import trange

# For Original RND Model
class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.predictor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

        self.target = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
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
        return (predict_feature - target_feature).pow(2).sum(1) 
        #만약 input = 2, output = 100, batchsize = 32라면 return시 (32)
        
    def reset_predictor(self):
        # predictor 네트워크의 가중치와 편향을 다시 초기화
        for p in self.predictor.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
    
# For Reward Predicting RND Model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self._initialize_weights()
    
    def forward(self, x):
        return self.MLP(x).squeeze(-1)
    
    def _initialize_weights(self):
        for layer in self.MLP:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.1)
                layer.bias.data.fill_(0.1)
                
class Ensemble(nn.Module):
    def __init__(self, input_dim, output_dim, num_models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList([MLP(input_dim, output_dim) for _ in range(num_models)])
            
    def forward(self, x):
        return torch.stack([model(x) for model in self.models], dim=0)
    
    
    
class ReplayBuffer():
    def __init__(self,
                 buffer_size,
                 device,
                 rnd_input = 2,
                 rnd_output = 100,
                 learning_rate = 1e-3,
                 exploration_mode = True,
                 rewards_type = 'mixed',
                 RND = False,
                 max_reward = True,
                 phaseB = 4000,
                 ) -> None:
        self.buffer_size = buffer_size
        self.device = device
        self.real_size = 0
        self.iter = 0
        self.intrinsic_clamp_val = 100
        self.exploration_mode = exploration_mode
        self.RND = RND
        self.phaseB = phaseB
        self.max_reward = max_reward
        
        # Initialize the buffer
        self.buffer = torch.tensor([]).cpu()
        self.true_rewards = torch.tensor([]).cpu()
        if self.max_reward == False:
            self.intrinsic_rewards = torch.tensor([]).cpu()
        
        if self.max_reward:
            self.max_true_reward = -100

        # Rewards Type
        self.rewards_type = rewards_type
        if self.rewards_type not in ["true", "intrinsic", "mixed"]:
            raise ValueError("Invalid rewards type. Choose from 'true', 'intrinsic', 'mixed'")
        
        # Initialize RND model for intrinsic reward
        if self.max_reward == False:
            if self.RND:
                self.rnd_model = RNDModel(rnd_input, rnd_output).to(device)
                self.rnd_optimizer = torch.optim.Adam(self.rnd_model.parameters(), lr=learning_rate)
            else:
                self.rnd_model = Ensemble(rnd_input, 1, 5).to(device)
                self.rnd_optimizer = torch.optim.Adam(self.rnd_model.parameters(), lr=learning_rate)
        
        
    def add(self, transition, rewards):
        self.iter += 1
        num_input_transitions = transition.shape[0]
        if self.real_size + num_input_transitions > self.buffer_size:
            self.truncate(num_input_transitions)
        
        # Reward Clamping to [-100, ]
        rewards = torch.clamp(rewards, min=-100)
        
        if self.max_reward == False:
            if self.iter != 1:
                if self.RND:
                    with torch.no_grad():
                        intrinsic_reward = self.rnd_model.predict_error(self.standardize_transition(transition))
                else:
                    with torch.no_grad():
                        pred = self.rnd_model(self.standardize_transition(transition))
                        intrinsic_reward = (pred - rewards).pow(2).mean(dim=0) 
            else:
                intrinsic_reward = torch.zeros_like(rewards)
        
        transition = transition.detach().cpu()
        rewards = rewards.detach().cpu()
        
        if self.max_reward == False:
            intrinsic_reward = intrinsic_reward.detach().cpu()
            
        if self.max_reward:
            self.max_true_reward = max(self.max_true_reward, max(rewards))
        
        self.buffer = torch.cat([self.buffer, transition], dim=0)
        self.true_rewards = torch.cat([self.true_rewards, rewards], dim=0)
        
        if self.max_reward == False:
            self.intrinsic_rewards = torch.cat([self.intrinsic_rewards, intrinsic_reward], dim=0)
        
        self.real_size += num_input_transitions
        
            
    def sample(self, batch_size, with_indices=False):
        indices = np.random.choice(self.real_size, batch_size, replace=False)    
        # Get samples
        samples = self.buffer[indices].to(self.device)
        true_rewards = self.true_rewards[indices].to(self.device)
        if self.max_reward == False:
            intrinsic_rewards = self.intrinsic_rewards[indices].to(self.device)
        
            updated_intrinsic_reward = self.train_rnd(self.standardize_transition(samples), true_rewards)
            # Update intrinsic rewards
            self.intrinsic_rewards[indices] = updated_intrinsic_reward.detach().cpu()
        
        if self.max_reward == False:
            rewards = self.get_rewards(true_rewards, intrinsic_rewards) 
        else:
            rewards = torch.tensor([self.max_true_reward] * batch_size).to(self.device)
        
        if with_indices: return samples, rewards, indices
        return samples, rewards
        

        
    def get_rewards(self, true_rewards, intrinsic_rewards):
        if self.rewards_type == "true":
            return true_rewards
        elif self.rewards_type == "intrinsic":
            return intrinsic_rewards
        else:
            # experiment 1
            #return true_rewards * (1-self.normalize_intrinsic_rewards(intrinsic_rewards))
            # experiment 2
            return true_rewards + intrinsic_rewards
            # experiment 3 #give maximum true rewards
            #return np.full_like(true_rewards, fill_value=self.max_true_reward, dtype=np.float32)
            
    # def normalize_intrinsic_rewards(self, intrinsic_rewards):
    #    return (intrinsic_rewards - self.min_intrinsic_reward) / (self.max_intrinsic_reward - self.min_intrinsic_reward)
    
    def standardize_transition(self, transition: torch.Tensor):
        standardized_transition = (transition - transition.mean(dim=0)) / transition.std(dim=0).clamp(min=-5, max=5)
        return standardized_transition
        
    def truncate(self, num_samples, return_samples=False):
        dropping_indices = self.get_probabilstic_indices(num_samples, method="low_true_reward")
        non_dropping_indices = torch.tensor([i for i in range(self.real_size) if i not in dropping_indices])
        # Dropping_indices = torch.tensor
        if return_samples:
            return self.buffer[dropping_indices]
        # Drop the lowest intrinsic reward samples
        self.buffer = self.buffer[non_dropping_indices]
        self.true_rewards = self.true_rewards[non_dropping_indices]
        if self.max_reward == False:
            self.intrinsic_rewards = self.intrinsic_rewards[non_dropping_indices]
        self.real_size = max(0, self.real_size - num_samples)  
    
    #----------------- For testing -----------------#
    def get_probabilstic_indices(self, num_samples, method="low_true_reward", decay = 0.99):
        #if method == "low_mixed_reward":
            # experiment 1
            #median_reward = np.median(self.true_rewards * (1-self.normalize_intrinsic_rewards(self.intrinsic_rewards)))
            #indices = np.where(np.array(self.true_rewards * (1-self.normalize_intrinsic_rewards(self.intrinsic_rewards))) < median_reward)[0]
            # experiment 2
            #median_reward = np.median(np.array(self.true_rewards) + np.array(self.intrinsic_rewards))
            #indices = np.where(np.array(self.true_rewards) + np.array(self.intrinsic_rewards) < median_reward)[0]
            # experiment 3
            #median_reward = np.median(np.log(np.exp(self.true_rewards) + self.intrinsic_rewards))
            #indices = np.where(np.array(np.log(np.exp(self.true_rewards) + self.intrinsic_rewards)) < median_reward)[0]
        #    if len(indices) < num_samples:
        #        indices = np.random.choice(self.real_size, num_samples, replace=False)
        #    else:
        #        indices = np.random.choice(indices, num_samples, replace=False)
        
        if method == "low_mixed_reward":
            indices = torch.argsort(self.true_rewards + self.intrinsic_rewards)[:num_samples]
        
        elif method == "low_true_reward":
            indices = torch.argsort(self.true_rewards)[:num_samples]
            
        elif method == "low_intrinsic_reward":
            indices = torch.argsort(self.intrinsic_rewards)[:num_samples]
        
        return indices      
        
    #----------------- RND Train/Update-----------------#
    def train_rnd(self, transition, rewards):
        if self.RND:
            self.rnd_optimizer.zero_grad()
            intrinsic_reward = self.rnd_model.predict_error(transition)
            loss = intrinsic_reward.mean()
            loss.backward()
            self.rnd_optimizer.step()

        else:
            self.rnd_optimizer.zero_grad()
            pred = self.rnd_model(transition)
            intrinsic_reward = (pred - rewards).pow(2).mean(dim=0)
            loss = intrinsic_reward.mean()
            loss.backward()
            self.rnd_optimizer.step()
        
        return intrinsic_reward
        
    #----------------- Utils -----------------#
        
    def __len__(self):
        return self.real_size