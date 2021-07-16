import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import wandb
import pickle
import random

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
        #print("\n\n---STATE---")
        #state = [item for sublist in state for item in sublist]  # flatten the state from a list of lists into a list
        #print(state)
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
        self.checkpoint_file = os.path.join(checkpoint_dir, 'actor_torch_ppo_3.1.pth')
        
        #--------------------------------------------------------
        self.num_actions = 20
        self.num_accepted = 30
        self.num_channels = self.num_actions+self.num_accepted+1
        self.embedding_dim = 50
        self.num_filters = 10

        with open("embedding_matrix.pkl", 'rb') as f:
            embedding_matrix = pickle.load(f)

        vocab_size = embedding_matrix.shape[0]
        print(f"vocab_size: {vocab_size}")
        vector_size = embedding_matrix.shape[1]
 
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        '''
        self.embedding = nn.Embedding(
            num_embeddings = 201585,
            embedding_dim = self.embedding_dim) 
            #padding_idx = 50, 
            #max_norm = 1)
        '''
        self.conv2d_0 = nn.Conv2d(
            in_channels = self.embedding_dim,
            out_channels = self.num_filters,
            kernel_size = (1, 1))
        self.conv2d_1 = nn.Conv2d(
            in_channels = self.embedding_dim,
            out_channels = self.num_filters,
            kernel_size = (2, 1))
        self.conv2d_2 = nn.Conv2d(
            in_channels = self.embedding_dim,
            out_channels = 1,
            kernel_size = (5, 1))
        self.conv2d_3 = nn.Conv2d(
            in_channels = self.embedding_dim,
            out_channels = self.num_filters,
            kernel_size = (10, 1))
        self.conv2d_4 = nn.Conv2d(
            in_channels = self.embedding_dim,
            out_channels = self.num_filters,
            kernel_size = (20, 1))
        self.ReLU = nn.ReLU()
        self.maxpool2d_0 = nn.MaxPool2d(
            kernel_size = (50, 1),
            stride = 1)
        self.maxpool1d_1 = nn.MaxPool1d(
            kernel_size = 49,
            stride = 1)
        self.maxpool1d_2 = nn.MaxPool1d(
            kernel_size = 46,
            stride = 1)
        self.maxpool1d_3 = nn.MaxPool1d(
            kernel_size = 41,
            stride = 1)
        self.maxpool1d_4 = nn.MaxPool1d(
            kernel_size = 31,
            stride = 1)
        self.linear_0 = nn.Linear(
            in_features = 5*self.num_channels*self.num_filters,
            #in_features = 195,
            out_features = 64)
        # ReLU
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.linear_1 = nn.Linear(
            in_features = 64,
            out_features = self.num_actions)
        self.softmax = nn.Softmax(3)
        

        if not torch.cuda.is_available(): # NOT sbagliato
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

        #self.optimizer = optim.Adam(self.parameters(), lr=alpha)


    def forward(self, input):
        #print("\n\n\n\n---FORWARD---\n")
        embedding_output = self.embedding(input)

        embedding_output_reshaped = embedding_output.reshape(input.shape[0], self.embedding_dim, 50, self.num_channels).detach().clone()

        conv2d_0_output = self.conv2d_0(embedding_output_reshaped)
        conv2d_0_output_relu = self.ReLU(conv2d_0_output)
        conv2d_1_output = self.conv2d_0(embedding_output_reshaped)
        conv2d_1_output_relu = self.ReLU(conv2d_1_output)
        conv2d_2_output = self.conv2d_0(embedding_output_reshaped)
        conv2d_2_output_relu = self.ReLU(conv2d_2_output)
        conv2d_3_output = self.conv2d_0(embedding_output_reshaped)
        conv2d_3_output_relu = self.ReLU(conv2d_3_output)
        conv2d_4_output = self.conv2d_0(embedding_output_reshaped)
        conv2d_4_output_relu = self.ReLU(conv2d_4_output)

        max_0_output = self.maxpool2d_0(conv2d_0_output_relu)
        max_1_output = self.maxpool2d_0(conv2d_1_output_relu)
        max_2_output = self.maxpool2d_0(conv2d_2_output_relu)
        max_3_output = self.maxpool2d_0(conv2d_3_output_relu)
        max_4_output = self.maxpool2d_0(conv2d_4_output_relu)

        linear_input = torch.cat((max_0_output, max_1_output, max_2_output,\
            max_3_output, max_4_output), dim = -1)

        if self.num_filters != 1:
            linear_input = linear_input.reshape(input.shape[0], 1, 1, 5*self.num_channels*self.num_filters).detach().clone()

        linear_0_output = self.linear_0(linear_input)
        linear_0_output_relu = self.ReLU(linear_0_output)
        linear_0_output_relu_dropout = self.dropout(linear_0_output_relu)
        linear_1_output = self.linear_1(linear_0_output_relu_dropout)

        #print(f"linear_1_output.shape: {linear_1_output.shape}")
        
        return self.softmax(linear_1_output)
    #------------------------------------------------------------

    def save_checkpoint(self):
        try: os.makedirs(self.checkpoint_dir)
        except FileExistsError: print("Directory " , self.checkpoint_dir ,  " already exists")
        print(self.state_dict())
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    def load_checkpoint_path(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))



class CriticNetwork(nn.Module):
    #def __init__(self, input_dims, alpha, checkpoint_dir='temp/ppo'):
    def __init__(self, alpha, checkpoint_dir='temp\\ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, 'critic_torch_ppo_3.1.pth')

        #self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        #--------------------------------------------------------
        self.num_actions = 20
        self.num_accepted = 30
        self.num_channels = self.num_actions+self.num_accepted+1
        self.embedding_dim = 50
        self.num_filters = 10

        with open("embedding_matrix.pkl", 'rb') as f:
            embedding_matrix = pickle.load(f)

        vocab_size = embedding_matrix.shape[0]
        vector_size = embedding_matrix.shape[1]
 
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        '''
        self.embedding = nn.Embedding(
            num_embeddings = 201585,
            embedding_dim = self.embedding_dim) 
            #padding_idx = 50, 
            #max_norm = 1)
        '''
        self.conv2d_0 = nn.Conv2d(
            in_channels = self.embedding_dim,
            out_channels = self.num_filters,
            kernel_size = (1, 1))
        self.conv2d_1 = nn.Conv2d(
            in_channels = self.embedding_dim,
            out_channels = self.num_filters,
            kernel_size = (2, 1))
        self.conv2d_2 = nn.Conv2d(
            in_channels = self.embedding_dim,
            out_channels = 1,
            kernel_size = (5, 1))
        self.conv2d_3 = nn.Conv2d(
            in_channels = self.embedding_dim,
            out_channels = self.num_filters,
            kernel_size = (10, 1))
        self.conv2d_4 = nn.Conv2d(
            in_channels = self.embedding_dim,
            out_channels = self.num_filters,
            kernel_size = (20, 1))
        self.ReLU = nn.ReLU()
        self.maxpool2d_0 = nn.MaxPool2d(
            kernel_size = (50, 1),
            stride = 1)
        self.maxpool1d_1 = nn.MaxPool1d(
            kernel_size = 49,
            stride = 1)
        self.maxpool1d_2 = nn.MaxPool1d(
            kernel_size = 46,
            stride = 1)
        self.maxpool1d_3 = nn.MaxPool1d(
            kernel_size = 41,
            stride = 1)
        self.maxpool1d_4 = nn.MaxPool1d(
            kernel_size = 31,
            stride = 1)
        self.linear_0 = nn.Linear(
            in_features = 5*self.num_channels*self.num_filters,
            #in_features = 195,
            out_features = 64)
        # ReLU
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.linear_1 = nn.Linear(
            in_features = 64,
            out_features = 1)


        self.device = torch.device('cuda:0' if not torch.cuda.is_available() else 'cpu')  # NOT sbagliato
        self.to(self.device)


    def forward(self, input):
        embedding_output = self.embedding(input)

        embedding_output_reshaped = embedding_output.reshape(input.shape[0], self.embedding_dim, 50, self.num_channels).detach().clone()

        conv2d_0_output = self.conv2d_0(embedding_output_reshaped)
        conv2d_0_output_relu = self.ReLU(conv2d_0_output)
        conv2d_1_output = self.conv2d_0(embedding_output_reshaped)
        conv2d_1_output_relu = self.ReLU(conv2d_1_output)
        conv2d_2_output = self.conv2d_0(embedding_output_reshaped)
        conv2d_2_output_relu = self.ReLU(conv2d_2_output)
        conv2d_3_output = self.conv2d_0(embedding_output_reshaped)
        conv2d_3_output_relu = self.ReLU(conv2d_3_output)
        conv2d_4_output = self.conv2d_0(embedding_output_reshaped)
        conv2d_4_output_relu = self.ReLU(conv2d_4_output)

        max_0_output = self.maxpool2d_0(conv2d_0_output_relu)
        max_1_output = self.maxpool2d_0(conv2d_1_output_relu)
        max_2_output = self.maxpool2d_0(conv2d_2_output_relu)
        max_3_output = self.maxpool2d_0(conv2d_3_output_relu)
        max_4_output = self.maxpool2d_0(conv2d_4_output_relu)

        linear_input = torch.cat((max_0_output, max_1_output, max_2_output,\
            max_3_output, max_4_output), dim = -1)

        if self.num_filters != 1:
            linear_input = linear_input.reshape(input.shape[0], 1, 1, 5*self.num_channels*self.num_filters).detach().clone()

        linear_0_output = self.linear_0(linear_input)
        linear_0_output_relu = self.ReLU(linear_0_output)
        linear_0_output_relu_dropout = self.dropout(linear_0_output_relu)
        linear_1_output = self.linear_1(linear_0_output_relu_dropout)
        
        return linear_1_output
    #------------------------------------------------------------

    def save_checkpoint(self):
        try: os.makedirs(self.checkpoint_dir)
        except FileExistsError: print("Directory " , self.checkpoint_dir ,  " already exists")
        #print(self.state_dict())
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    def load_checkpoint_path(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

class Agent:
    #def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            #policy_clip=0.2, batch_size=64, n_epochs=10):
    def __init__(self, gamma=0.7, alpha=0.0003, gae_lambda=0.7,
            policy_clip=0.4, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        #self.actor = ActorNetwork(n_actions, input_dims, alpha)
        #self.critic = CriticNetwork(input_dims, alpha)
        self.actor = ActorNetwork(alpha)
        #self.actor.optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        print("PAREMETERS")
        for name, param in self.actor.named_parameters():
            if name == 'embedding.weight':
                param.requires_grad = False
            if param.requires_grad:
                print(name, param.data)

        #self.actor.optimizer = optim.RMSprop(self.actor.parameters())
        self.actor.optimizer = optim.Adam(self.actor.parameters())
        self.critic = CriticNetwork(alpha)
        self.critic.optimizer = optim.Adam(self.critic.parameters())
        #self.critic.optimizer = optim.RMSprop(self.critic.parameters())
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

    def load_models_path(self, checkpoint_file):
        print('... loading models ...')
        self.actor.load_checkpoint_path("actor_"+checkpoint_file)
        self.critic.load_checkpoint_path("critic_"+checkpoint_file)

    def choose_action(self, observation):
        #print("\n---OBSERVATION---")
        #observation = observation.astype('int64')
        #print(np.array(observation))
        state = torch.tensor(observation).long().to(self.actor.device)
        #state = state.unsqueeze_(0)
        #state = state.permute(0,3,1,2)
        #print("\n---SHAPE---")
        #print(np.shape(state))

        #print(np.shape(state.unsqueeze(0)))
        #print(state.unsqueeze(0))
        #print(state.unsqueeze(0).shape)
        dist = self.actor(state.unsqueeze(0))
        value = self.critic(state.unsqueeze(0))

        randomNumber = random.randint(0, 49)
        if randomNumber == 0:
            print(f"Action distribution: {dist}")

        dist = Categorical(dist)
        
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        #print(probs)
        action = torch.squeeze(action).item()
        #print(action)
        value = torch.squeeze(value).item()
        #print(value)

        return action, probs, value

    def learn(self):
        torch.autograd.set_detect_anomaly(True)
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            #print(self.memory.generate_batches())

            values = vals_arr
            #print(action_arr)
            #print(values)
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
            #print(advantage)

            values = torch.tensor(values).to(self.actor.device)
            #print(f"BATCHES: {batches}")
            for batch in batches:
                #print("ENTRATO")
                states = torch.tensor(state_arr[batch], dtype=torch.int64).to(self.actor.device)
                #print(states.shape)
                #print(state_arr)
                #print(batch)
                #print(state_arr[batch])
                #states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                #print(old_probs)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)
                #print(actions)

                dist = self.actor(states)
                #print("ACTOR_DIST: ", dist)
                dist = Categorical(dist)                
                
                critic_value = self.critic(states)
                #print("CRITIC_VALUE: ", critic_value)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                #print(advantage[batch])
                #print(values[batch])
                #print(returns)
                #print("\n+++Returns and Critic_value+++")
                #print(returns, critic_value)
                critic_loss = (returns-critic_value)**2
                #print("\n+++CRITIC LOSSES+++")
                #print(critic_loss)
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 2*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        print("\n---LOSSES---")
        print(f"actor: {actor_loss}")
        print(f"critic*2: {critic_loss*2}")
        print(f"total: {total_loss}")

        wandb.log({'Train actor_loss': actor_loss, 'Train critic_loss*2': critic_loss*2, 'Train total_loss': total_loss})

        self.memory.clear_memory()


