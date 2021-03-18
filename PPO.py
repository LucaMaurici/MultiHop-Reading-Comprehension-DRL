import os
import numpy as no
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.state = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
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
    def __init__(self, alpha, checkpoint_dir='temp/ppo'):
        
        super(ActorNetwork, self).__init__()

        
        self.checkpoint_file = os.path.join(checkpoint_dir, 'actor_torch_ppo')
        
        #self.actor = nn.Sequential(nn.Linear(*input_dims, 256),nn.ReLU(),nn.Linear(256, 128),nn.ReLU(),nn.Linear(128, n_action),nn.Softmax(dim=-1))
        
        #--------------------------------------------------------
        self.num_actions = 8
        self.num_channels = self.num_actions+30+1
        self.num_accepted = 30

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
        self.embedding = nn.Embedding(
            num_embeddings = 87429, 
            embedding_dim = 50, 
            padding_idx = 50, 
            max_norm = 1)
        self.conv2d = nn.Conv2d(
            in_channels = self.num_channels,
            out_channels = self.num_channels,
            kernel_size = (50, 1, 1))
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


        
        
        if T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')
        self.to(self.device)

        #self.optimizer = optim.Adam(self.parameters(), lr=alpha)


    def forward(self, input):
        embedding_output = self.embedding(input)
        conv2d_output = self.conv2d(embedding_output)
        conv2d_output = f.relu_(conv2d_output)
        conv2d_1_output = self.conv2d_1(embedding_output)
        conv2d_1_output = f.relu_(conv2d_1_output)
        conv2d_2_output = self.conv2d_2(embedding_output)
        conv2d_2_output = f.relu_(conv2d_2_output)
        conv2d_3_output = self.conv2d_3(embedding_output)
        conv2d_4_output = self.conv2d_4(embedding_output)
        conv2d_4_output = f.relu_(conv2d_4_output)
        maxpool1d_input = torch.cat((conv2d_output, conv2d_1_output, conv2d_2_output, conv2d_3_output, conv2d_4_output), dim=0)
        maxpool1d_output = self.maxpool1d(maxpool1d_input)
        linear_output = self.linear(maxpool1d_output)
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
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))



class CriticNetwork(nn.Module):
    #def __init__(self, input_dims, alpha, checkpoint_dir='tmp/ppo'):
    def __init__(self, alpha, checkpoint_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, 'critic_torch_ppo')
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
        self.num_channels = self.num_actions+30+1
        self.num_accepted = 30

        self.embedding = nn.Embedding(
            num_embeddings = 87429, 
            embedding_dim = 50, 
            padding_idx = 50, 
            max_norm = 1)
        self.conv2d = nn.Conv2d(
            in_channels = self.num_channels, 
            out_channels = self.num_channels, 
            kernel_size = (50, 1, 1))
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
            out_features = 1)


        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, input):
        embedding_output = self.embedding(input)
        conv2d_output = self.conv2d(embedding_output)
        conv2d_output = f.relu_(conv2d_output)
        conv2d_1_output = self.conv2d_1(embedding_output)
        conv2d_1_output = f.relu_(conv2d_1_output)
        conv2d_2_output = self.conv2d_2(embedding_output)
        conv2d_2_output = f.relu_(conv2d_2_output)
        conv2d_3_output = self.conv2d_3(embedding_output)
        conv2d_4_output = self.conv2d_4(embedding_output)
        conv2d_4_output = f.relu_(conv2d_4_output)
        maxpool1d_input = torch.cat((conv2d_output, conv2d_1_output, conv2d_2_output, conv2d_3_output, conv2d_4_output), dim=0)
        maxpool1d_output = self.maxpool1d(maxpool1d_input)
        linear_output = self.linear(maxpool1d_output)
        
        return linear_output
    #------------------------------------------------------------

    '''
    def forward(self, state):
        return self.critic(state)
    '''

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

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
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

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
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

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


