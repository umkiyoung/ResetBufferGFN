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
                 device = "cpu",
                 input_dim=2, 
                 hidden_dim=128,
                 output_dim=1, 
                 num_models=2,
                 decay=10000,
                 alpha = 1,
                 exploration_mode = True,
                 ):
        
        self.max_size = max_size
        self.decay = decay
        self.alpha = alpha        
        self.device = device
        
        self.buffer = []
        self.rewards = []
        self.novelty = []
        self.novelty_threshold = 0
        self.iter = 0
        self.exploration_mode = exploration_mode
        
        self.trainer = Trainer(
            Ensemble(input_dim, hidden_dim, output_dim, num_models),
            optimizer_cls=torch.optim.Adam,
            criterion=nn.MSELoss()
        )
        self.trainer.model.to(self.device)
            
    def add(self, transition, rewards):
        self.iter += 1
        
        # Truncate Buffer
        if len(self.buffer) + transition.shape[0] >= self.max_size:
            self.low_reward_truncate(transition.shape[0])
            
        # Add Transition
        y_pred = self.trainer.model(transition)
        max_novelty = torch.max((y_pred - rewards.unsqueeze(0))**2, dim=0)[0]
        self.trainer.train(transition, rewards.unsqueeze(0))
        transition, rewards = transition.cpu().detach().numpy(), rewards.cpu().detach().numpy()
        max_novelty = max_novelty.cpu().detach().numpy()
        
        for i in range(transition.shape[0]):
            if max_novelty[i] <= self.novelty_threshold:
                continue
            self.buffer.append(transition[i])
            self.rewards.append(max(rewards[i], -100))
            self.novelty.append(max_novelty[i])
                        
    def low_reward_truncate(self, num_samples, return_samples=False):
        criteria = np.array(self.rewards)
        indices = np.argsort(criteria)[:num_samples]
        if return_samples:
            samples = [self.buffer[idx] for idx in indices]
            return torch.tensor(np.array(samples)).float().to(self.device)
        
        self.buffer = [self.buffer[i] for i in range(len(self.buffer)) if i not in indices]
        self.rewards = [self.rewards[i] for i in range(len(self.rewards)) if i not in indices]
        self.novelty = [self.novelty[i] for i in range(len(self.novelty)) if i not in indices]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sample(self, batch_size):
        priorities = self.get_priorities()
        priorities = 0.9999 * priorities + 0.0001 / len(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=priorities, replace=False)
        
        samples = [self.buffer[idx] for idx in indices]
        rewards  = [self.rewards[idx] for idx in indices]
        
        # Update Novelty
        samples_tensor, rewards_tensor = torch.tensor(np.array(samples)).float().to(self.device), torch.tensor(np.array(rewards)).float().to(self.device)
        self.trainer.train(samples_tensor, rewards_tensor)
        y_pred = self.trainer.model(samples_tensor)
        max_novelty = torch.max((y_pred - rewards_tensor.unsqueeze(0))**2, dim=0)[0]
        for i, idx in enumerate(indices):
            self.novelty[idx] = max_novelty[i].cpu().detach().numpy()       

        return samples_tensor, rewards_tensor

    def get_priorities(self):
        if self.exploration_mode == True:
            normalized_novelty = self.normalize(self.novelty)
            normalized_rewards = self.normalize(self.rewards)
            priorities = np.array(normalized_novelty) * self.alpha * max(0,(self.decay - self.iter / self.decay))
            priorities = priorities + np.array(normalized_rewards)
            priorities = self.softmax(priorities)    
        else:
            normalized_rewards = self.normalize(self.rewards)
            priorities = self.softmax(self.rewards)
        return priorities
            
    def normalize(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    def __len__(self):
        return len(self.buffer)
