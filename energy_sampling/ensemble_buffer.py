import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList([MLP(input_dim, hidden_dim, output_dim) for _ in range(num_models)])
            
    def forward(self, x):
        return torch.stack([model(x) for model in self.models], dim=0)
    
class Trainer():
    def __init__(self, model, optimizer_cls, criterion):
        self.model = model
        self.ensemble_size = len(model.models)
        self.optimizer = optimizer_cls(self.model.parameters())
        self.criterion = criterion
        
    def train(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        y_pred = torch.mean(y_pred, dim=0)
        # Ensemble Loss
        loss = self.criterion(y_pred, y.squeeze())
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def eval(self, x, y):
        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
        return loss.item()

# Ensemble Based
class ReplayBuffer():
    def __init__(self, 
                 max_size, 
                 device="cpu",
                 input_dim=2, 
                 hidden_dim=128,
                 output_dim=1, 
                 num_models=2,
                 decay=10000,
                 alpha=1,
                 exploration_mode=True):
        
        self.max_size = max_size
        self.decay = decay
        self.alpha = alpha        
        self.device = device
        
        self.buffer = np.empty((max_size, input_dim))
        self.rewards = np.empty(max_size)
        self.novelty = np.empty(max_size)
        self.iter = 0
        self.exploration_mode = exploration_mode
        self.current_size = 0
        
        self.trainer = Trainer(
            Ensemble(input_dim, hidden_dim, output_dim, num_models),
            optimizer_cls=torch.optim.Adam,
            criterion=nn.MSELoss()
        )
        self.trainer.model.to(self.device)
            
    def add(self, transition, rewards):
        self.iter += 1
        num_new_samples = transition.shape[0]
        
        # Truncate Buffer
        if self.current_size + num_new_samples > self.max_size:
            self.low_reward_truncate(num_new_samples)
        
        # Predict novelty and train model
        y_pred = self.trainer.model(transition)
        max_novelty = torch.max((y_pred - rewards.unsqueeze(0))**2, dim=0)[0]
        self.trainer.train(transition, rewards.unsqueeze(0))
        
        # Convert to numpy for storage
        transition, rewards = transition.cpu().detach().numpy(), rewards.cpu().detach().numpy()
        max_novelty = max_novelty.cpu().detach().numpy()
        
        # Add valid samples to buffer
        for i in range(num_new_samples):          
            if self.current_size < self.max_size:
                self.buffer[self.current_size] = transition[i]
                self.rewards[self.current_size] = max(rewards[i], -100)
                self.novelty[self.current_size] = max_novelty[i]
                self.current_size += 1
        
        self.update_all_novelty()
                        
    def low_reward_truncate(self, num_samples, return_samples=False):
        indices = np.argsort(self.rewards)[:num_samples]
        
        if return_samples:
            samples = self.buffer[indices]
            return torch.tensor(samples).float().to(self.device)
        
        mask = np.ones(self.current_size, dtype=bool)
        mask[indices] = False
        
        self.buffer = self.buffer[mask]
        self.rewards = self.rewards[mask]
        self.novelty = self.novelty[mask]
        self.current_size -= num_samples

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sample(self, batch_size):
        priorities = self.get_priorities()
        priorities = 0.9999 * priorities + 0.0001 / len(priorities)
        indices = np.random.choice(self.current_size, batch_size, p=priorities, replace=False)
        
        samples = self.buffer[indices]
        rewards  = self.rewards[indices]
        
        # Update Novelty
        samples_tensor = torch.tensor(samples).float().to(self.device)
        rewards_tensor = torch.tensor(rewards).float().to(self.device)
        self.trainer.train(samples_tensor, rewards_tensor)
        self.update_all_novelty()
        return samples_tensor, rewards_tensor
    
    def update_all_novelty(self):
        samples_tensor = torch.tensor(self.buffer[:self.current_size]).float().to(self.device)
        rewards_tensor = torch.tensor(self.rewards[:self.current_size]).float().to(self.device)
        y_pred = self.trainer.model(samples_tensor)
        max_novelty = torch.max((y_pred - rewards_tensor.unsqueeze(0))**2, dim=0)[0]
        self.novelty[:self.current_size] = max_novelty.cpu().detach().numpy()

    def get_priorities(self):
        if self.exploration_mode:
            normalized_novelty = self.normalize(self.novelty[:self.current_size])
            normalized_rewards = self.normalize(self.rewards[:self.current_size])
            priorities = normalized_novelty * self.alpha * max(0, (self.decay - self.iter / self.decay))
            priorities = priorities + normalized_rewards
            priorities = self.softmax(priorities)
        else:
            normalized_rewards = self.normalize(self.rewards[:self.current_size])
            priorities = self.softmax(normalized_rewards)
        return priorities
            
    def normalize(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    def __len__(self):
        return self.current_size
