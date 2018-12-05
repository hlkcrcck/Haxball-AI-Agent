import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from PIL import Image
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        """self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)"""
        self.ll1 = nn.Linear(6, 128)
        self.ll2 = nn.Linear(128,256)
        self.ll3 = nn.Linear(256, 64)
        self.ll4 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.dropout(F.relu(self.ll1(x)), 0.2)
        x = F.dropout(F.relu(self.ll2(x)), 0.2)
        x = F.dropout(F.relu(self.ll3(x)), 0.2)
        x = self.ll4(x)
        return x
        """x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))"""

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.1
EPS_DECAY = 4000
TARGET_UPDATE = 10

policy_net = DQN().double().to(device)
target_net = DQN().double().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.0001, weight_decay=0.99)
memory = ReplayMemory(100000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(0)[1].view(1, 1).cuda()
    else:
        return torch.LongTensor([[random.randrange(5)]]).cuda()

episode_durations = []

def plot_rewards():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    plt.title('Training... Eps:' + str(eps_threshold))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device).double()
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.double().cuda()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def train_P1(room,num_episodes):
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        room.reset()
        #last_screen = torch.from_numpy(np.array(room.get_screen())).double().cuda()
        current_screen = torch.from_numpy(np.array(room.get_screen())).double().cuda()
        #state = current_screen - last_screen
        state = current_screen
        cumulative_reward = 0
        for t in count():
            # Select and perform an action
            action = select_action(state)
            reward, done = room.step(action.item())
            cumulative_reward += reward
            reward = torch.Tensor([reward], device=device)

            # Observe new state
            #last_screen = current_screen
            current_screen = torch.from_numpy(np.array(room.get_screen())).double().cuda()
            if not done:
                #next_state = current_screen - last_screen
                next_state = current_screen
            else:
                next_state = None

            # Store the transition in memory
            if i_episode > 25:
                if not (state[2].item() == 0 and state[3].item() == 0):
                    prob = random.random()
                    if prob >= 0.5:
                        memory.push(state, action, next_state, reward)
            else:
                memory.push(state, action, next_state, reward)
            #memory.push(state, action, next_state, reward)    
            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                #ai.episode_durations.append(t + 1)
                episode_durations.append(cumulative_reward)
                plot_rewards()
                break
        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    plt.ioff()
    plt.show()
