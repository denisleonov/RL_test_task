import random
import numpy as np
import os
from train import Actor
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent:
    def __init__(self, state_dim, action_dim):
         self.actor = Actor(state_dim, action_dim)
         self.actor.load_state_dict(torch.load('actor.pkl'))
         self.actor.eval()
         self.actor.to(device)

        
    def act(self, state):
        state = np.array(state, dtype=np.float)
        state = torch.tensor(state).to(device, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).detach()

        return action.cpu().data.numpy()

