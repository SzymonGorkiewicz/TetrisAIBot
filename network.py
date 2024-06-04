import numpy as np
import torch
import random
from collections import deque
import torch.optim as optim
import torch.nn.functional as F

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_layer, output_layer): 
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_layer, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 128)
        self.fc5 = torch.nn.Linear(128, 128)
        self.fc6 = torch.nn.Linear(128, 128)
        self.fc7 = torch.nn.Linear(128, output_layer)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x

BUFFER_SIZE = 50_000
BATCH_SIZE = 32


class Memory():
    def __init__(self, policy_model, target_model, gamma, learning_rate):
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.batch_size = BATCH_SIZE
        self.policy_model = policy_model
        self.target_model = target_model
        self.gamma = gamma
        self.loss_function = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(policy_model.parameters(), lr=self.learning_rate)
        self.loss = 0

    def update_memory(self,state,action,reward,new_state,done):
        self.memory.append((state,action,reward,new_state,done))

    def sample(self):
        return random.sample(self.memory, BATCH_SIZE)
    
    def train_from_memory(self, sample):
        states, actions, rewards, new_states, dones = zip(*sample)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        new_states = torch.tensor(new_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

       
        
        q_values_policy = self.policy_model(states).squeeze()
        q_values_target = self.target_model(new_states).squeeze().detach()

        targets_values = q_values_policy.clone().detach()


        for i in range(len(sample)):
            if dones[i]:
                targets_values[i] = rewards[i]
            else:
                targets_values[i] = rewards[i] + self.gamma * q_values_target[i]
        
        q_values_policy = q_values_policy.view(-1)
        targets_values = targets_values.view(-1)

        loss = self.loss_function(q_values_policy, targets_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss.item()

        
    def save_checkpoint(self, filepath):
        checkpoint = {
            'policy_model_state_dict': self.policy_model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': list(self.memory),
            'gamma': self.gamma,
            'learning_rate': self.learning_rate
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_model.load_state_dict(checkpoint['policy_model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = deque(checkpoint['memory'], maxlen=BUFFER_SIZE)
        self.gamma = checkpoint['gamma']
        self.learning_rate = checkpoint['learning_rate']
        print(f"Checkpoint loaded from {filepath}")