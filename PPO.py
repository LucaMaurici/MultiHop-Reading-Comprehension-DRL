import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        #indices = np.arange(n_states, dtype=np.int64)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        print("\n\n---STATE---")
        #state = [item for sublist in state for item in sublist]  # flatten the state from a list of lists into a list
        print(state)
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []



class ActorNetwork(nn.Module):

    #def __init__(self, n_action, input_dims, alpha, checkpoint_dir='temp/ppo'):
    def __init__(self, alpha, checkpoint_dir='temp\\ppo'):
        
        super(ActorNetwork, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, 'actor_torch_ppo.pth')
        
        #self.actor = nn.Sequential(nn.Linear(*input_dims, 256),nn.ReLU(),nn.Linear(256, 128),nn.ReLU(),nn.Linear(128, n_action),nn.Softmax(dim=-1))
        
        #--------------------------------------------------------
        self.num_actions = 8
        self.num_accepted = 30
        self.num_channels = self.num_actions+self.num_accepted+1
        

        '''
        self.embedding = nn.Embedding(
            num_embeddings = 87429, 
            embedding_dim = 50, 
            padding_idx = 50, 
            max_norm = 1)
        self.conv2d = nn.Conv2d(
            in_channels = (50, 50, self.num_channels), 
            out_channels = (1, 50, self.num_channels), 
            kernel_size = (50, 1, 1))
        self.conv2d_1 = nn.Conv2d(
            in_channels = (50, 50, self.num_channels), 
            out_channels = (1, 48, self.num_channels), 
            kernel_size = (50, 2, 1))
        self.conv2d_2 = nn.Conv2d(
            in_channels = (50, 50, self.num_channels), 
            out_channels = (1, 48, self.num_channels), 
            kernel_size = (50, 5, 1))
        self.conv2d_3 = nn.Conv2d(
            in_channels = (50, 50, self.num_channels), 
            out_channels = (1, 48, self.num_channels), 
            kernel_size = (50, 10, 1))
        self.conv2d_4 = nn.Conv2d(
            in_channels = (50, 50, self.num_channels), 
            out_channels = (1, 48, self.num_channels), 
            kernel_size = (50, 20, 1))
        self.maxpool1d = nn.MaxPool1d(
            kernel_size = (1, 50, 1), 
            stride = 1)
        self.linear = nn.Linear(
            in_features = self.num_channels, 
            out_features = 8)
        self.softmax = nn.Softmax(
            name = Softmax, 
            dim = self.num_actions)
        '''
        '''
        self.embedding = nn.Embedding(
            num_embeddings = 87429, 
            embedding_dim = 50, 
            #padding_idx = 50, 
            max_norm = 1)
        self.conv2d = nn.Conv2d(
            in_channels = self.num_channels,
            out_channels = self.num_channels,
            kernel_size = (1, 1, 50),
            #stride = [1,1,0],
            #padding = [0,1,0],
            #dilation = [1,1,1]
            )
        self.conv2d_1 = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = self.num_channels,
            kernel_size = (50, 2, 1))
        self.conv2d_2 = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = self.num_channels,
            kernel_size = (50, 5, 1))
        self.conv2d_3 = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = self.num_channels,
            kernel_size = (50, 10, 1))
        self.conv2d_4 = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = self.num_channels,
            kernel_size = (50, 20, 1))
        self.maxpool1d = nn.MaxPool1d(
            kernel_size = (1, 50, 1), 
            stride = 1)
        self.linear = nn.Linear(
            in_features = self.num_channels, 
            out_features = 8)
        self.softmax = nn.Softmax(
            dim = self.num_actions)
        '''
        self.embedding = nn.Embedding(
            num_embeddings = 87429,
            embedding_dim = 51) 
            #padding_idx = 50, 
            #max_norm = 1)
        self.conv2d_0 = nn.Conv2d(
            in_channels = self.num_channels,
            out_channels = 1,
            kernel_size = (1, 51))
        self.conv2d_1 = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = 1,
            kernel_size = (2, 51))
        self.conv2d_2 = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = 1,
            kernel_size = (5, 51))
        self.conv2d_3 = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = 1,
            kernel_size = (10, 51))
        self.conv2d_4 = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = 1,
            kernel_size = (20, 51))
        #self.maxpool1d = nn.MaxPool1d(
            #kernel_size = (), 
            #stride = 1)
        self.linear = nn.Linear(
            in_features = 5,
            out_features = 8)
        self.softmax = nn.Softmax(0)
        

        
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

        #self.optimizer = optim.Adam(self.parameters(), lr=alpha)


    def forward(self, input):
        #print("\n---INPUT---\n")
        embedding_output = self.embedding(input)
        
        conv2d_0_output = self.conv2d_0(embedding_output)
        conv2d_0_output = f.relu_(conv2d_0_output)
        conv2d_1_output = self.conv2d_1(embedding_output)
        conv2d_1_output = f.relu_(conv2d_1_output)
        conv2d_2_output = self.conv2d_2(embedding_output)
        conv2d_2_output = f.relu_(conv2d_2_output)
        conv2d_3_output = self.conv2d_3(embedding_output)
        conv2d_3_output = f.relu_(conv2d_3_output)
        conv2d_4_output = self.conv2d_4(embedding_output)
        conv2d_4_output = f.relu_(conv2d_4_output)

        max_0_output = torch.max(conv2d_0_output)
        max_1_output = torch.max(conv2d_1_output)
        max_2_output = torch.max(conv2d_2_output)
        max_3_output = torch.max(conv2d_3_output)
        max_4_output = torch.max(conv2d_4_output)

        #print(max_0_output)
        #print(max_0_output.shape)

        linear_input = torch.tensor([max_0_output, max_1_output, max_2_output, max_3_output, max_4_output])

        #linear_input = torch.cat((max_0_output, max_1_output, max_2_output, max_3_output, max_4_output), dim=-1)
        #maxpool1d_output = self.maxpool1d(maxpool1d_input)
        linear_output = self.linear(linear_input)
        linear_output = f.relu_(linear_output)
        softmax_output = self.softmax(linear_output)
        
        return softmax_output
    #------------------------------------------------------------

    '''
    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist
    '''

    def save_checkpoint(self):
        try: os.makedirs(self.checkpoint_dir)
        except FileExistsError: print("Directory " , self.checkpoint_dir ,  " already exists")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))



class CriticNetwork(nn.Module):
    #def __init__(self, input_dims, alpha, checkpoint_dir='temp/ppo'):
    def __init__(self, alpha, checkpoint_dir='temp\\ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, 'critic_torch_ppo.pth')
        '''
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
        )
        '''

        #self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        

        #--------------------------------------------------------
        self.num_actions = 8
        self.num_accepted = 30
        self.num_channels = self.num_actions+self.num_accepted+1


        self.embedding = nn.Embedding(
            num_embeddings = 87429,
            embedding_dim = 51)
            #padding_idx = 50, 
            #max_norm = 1)
        self.conv2d_0 = nn.Conv2d(
            in_channels = self.num_channels,
            out_channels = 1,
            kernel_size = (1, 51))
        self.conv2d_1 = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = 1,
            kernel_size = (2, 51))
        self.conv2d_2 = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = 1,
            kernel_size = (5, 51))
        self.conv2d_3 = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = 1,
            kernel_size = (10, 51))
        self.conv2d_4 = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = 1,
            kernel_size = (20, 51))
        #self.maxpool1d = nn.MaxPool1d(
            #kernel_size = (), 
            #stride = 1)
        self.linear = nn.Linear(
            in_features = 5,
            out_features = 1)


        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, input):
        embedding_output = self.embedding(input)
        
        conv2d_0_output = self.conv2d_0(embedding_output)
        conv2d_0_output = f.relu_(conv2d_0_output)
        conv2d_1_output = self.conv2d_1(embedding_output)
        conv2d_1_output = f.relu_(conv2d_1_output)
        conv2d_2_output = self.conv2d_2(embedding_output)
        conv2d_2_output = f.relu_(conv2d_2_output)
        conv2d_3_output = self.conv2d_3(embedding_output)
        conv2d_3_output = f.relu_(conv2d_3_output)
        conv2d_4_output = self.conv2d_4(embedding_output)
        conv2d_4_output = f.relu_(conv2d_4_output)

        max_0_output = torch.max(conv2d_0_output)
        max_1_output = torch.max(conv2d_1_output)
        max_2_output = torch.max(conv2d_2_output)
        max_3_output = torch.max(conv2d_3_output)
        max_4_output = torch.max(conv2d_4_output)

        linear_input = torch.tensor([max_0_output, max_1_output, max_2_output, max_3_output, max_4_output])

        #linear_input = torch.cat((max_0_output, max_1_output, max_2_output, max_3_output, max_4_output), dim=-1)
        #maxpool1d_output = self.maxpool1d(maxpool1d_input)
        linear_output = self.linear(linear_input)
        linear_output = f.relu_(linear_output)
        
        return linear_output
    #------------------------------------------------------------

    '''
    def forward(self, state):
        return self.critic(state)
    '''

    def save_checkpoint(self):
        try: os.makedirs(self.checkpoint_dir)
        except FileExistsError: print("Directory " , self.checkpoint_dir ,  " already exists")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    #def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            #policy_clip=0.2, batch_size=64, n_epochs=10):
    def __init__(self, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        #self.actor = ActorNetwork(n_actions, input_dims, alpha)
        #self.critic = CriticNetwork(input_dims, alpha)
        self.actor = ActorNetwork(alpha)
        self.actor.optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        #self.optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic = CriticNetwork(alpha)
        self.critic.optimizer = optim.Adam(self.critic.parameters(), lr=alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        print("\n---OBSERVATION---")
        #observation = observation.astype('int64')
        #print(np.array(observation))
        state = torch.tensor(observation).to(self.actor.device).long()
        #state = state.unsqueeze_(0)
        #state = state.permute(0,3,1,2)
        print("\n---SHAPE---")
        print(np.shape(state))

        dist = self.actor(state.unsqueeze(0))
        value = self.critic(state.unsqueeze(0))
        print(dist)
        dist = Categorical(dist)
        
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        print(probs)
        action = torch.squeeze(action).item()
        print(action)
        value = torch.squeeze(value).item()
        print(value)

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.int64).to(self.actor.device)
                print(state_arr)
                print(batch)
                print(state_arr[batch])
                #states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                dist = Categorical(dist)
                
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


