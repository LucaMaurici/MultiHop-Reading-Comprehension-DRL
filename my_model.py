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
from typing import Optional

class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.outputs = []
        self.ground_truths = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.outputs),\
                np.array(self.ground_truths),\
                batches

    def store_memory(self, state, output, ground_truth):
        #print("\n\n---STATE---")
        #state = [item for sublist in state for item in sublist]  # flatten the state from a list of lists into a list
        #print(state)
        self.states.append(state)
        self.outputs.append(output)
        self.ground_truths.append(ground_truth)

    def clear_memory(self):
        self.states = []
        self.outputs = []
        self.ground_truths = []



class ActorNetwork(nn.Module):

    '''
    def __init__(self, checkpoint_dir='temp\\ppo'):
        
        super(ActorNetwork, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, 'actor_torch_myModel_1.0.pth')
        
        #--------------------------------------------------------
        self.num_actions = 31
        self.num_accepted = 30
        self.num_channels = self.num_actions+self.num_accepted+1
        self.embedding_dim = 50
        self.num_filters = 1

        with open("embedding_matrix.pkl", 'rb') as f:
            embedding_matrix = pickle.load(f)

        vocab_size = embedding_matrix.shape[0]
        print(f"vocab_size: {vocab_size}")
        vector_size = embedding_matrix.shape[1]
 
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_size, padding_idx = 50)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.linear_1 = nn.Linear(
            in_features = self.embedding_dim * self.num_channels * 50,
            out_features = self.num_actions)
        #self.softmax = nn.Softmax(3)
        
        self.loss_fn = torch.nn.MSELoss(reduce=True)

        if not torch.cuda.is_available(): # NOT sbagliato
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)


    def forward(self, input, y: Optional[torch.Tensor] = None):
        #print("\n\n\n\n---FORWARD---\n")
        embedding_output = self.embedding(input)

        embedding_output_reshaped = embedding_output.reshape(input.shape[0], self.embedding_dim * self.num_channels * 50).detach().clone()

        linear_1_output = self.linear_1(embedding_output_reshaped).squeeze()


        result = {'pred': linear_1_output}
        # compute loss
        if y is not None:
            loss = self.loss(linear_1_output, y)
            result['loss'] = loss
        
        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)

    '''

    
    def __init__(self, checkpoint_dir='temp\\ppo'):
        
        super(ActorNetwork, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, 'actor_torch_myModel_1.1.pth')
        
        #--------------------------------------------------------
        self.num_actions = 31
        self.num_accepted = 30
        self.num_channels = self.num_actions+self.num_accepted+1
        self.embedding_dim = 50
        self.num_filters = 1000

        with open("embedding_matrix.pkl", 'rb') as f:
            embedding_matrix = pickle.load(f)

        vocab_size = embedding_matrix.shape[0]
        print(f"vocab_size: {vocab_size}")
        vector_size = embedding_matrix.shape[1]
 
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_size, padding_idx = 50)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

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
            out_channels = self.num_filters,
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
        self.dropoutConv = nn.Dropout(p=0.4, inplace=False)
        self.maxpool2d_0 = nn.MaxPool2d(
            kernel_size = (50, 1),
            stride = 1)
        self.maxpool2d_1 = nn.MaxPool2d(
            kernel_size = (49, 1),
            stride = 1)
        self.maxpool2d_2 = nn.MaxPool2d(
            kernel_size = (46, 1),
            stride = 1)
        self.maxpool2d_3 = nn.MaxPool2d(
            kernel_size = (41, 1),
            stride = 1)
        self.maxpool2d_4 = nn.MaxPool2d(
            kernel_size = (31, 1),
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
        #self.softmax = nn.Softmax(3)
        
        self.loss_fn = torch.nn.MSELoss(reduce=True)

        if torch.cuda.is_available(): # NOT sbagliato
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    
    def forward(self, input, y: Optional[torch.Tensor] = None):
        #print("\n\n\n\n---FORWARD---\n")
        embedding_output = self.embedding(input)

        embedding_output_reshaped = embedding_output.reshape(input.shape[0], self.embedding_dim, 50, self.num_channels).detach().clone()

        conv2d_0_output = self.conv2d_0(embedding_output_reshaped)
        conv2d_0_output_relu = self.ReLU(conv2d_0_output)
        conv2d_0_output_relu = self.dropoutConv(conv2d_0_output_relu)
        conv2d_1_output = self.conv2d_1(embedding_output_reshaped)
        conv2d_1_output_relu = self.ReLU(conv2d_1_output)
        conv2d_1_output_relu = self.dropoutConv(conv2d_1_output_relu)
        conv2d_2_output = self.conv2d_2(embedding_output_reshaped)
        conv2d_2_output_relu = self.ReLU(conv2d_2_output)
        conv2d_2_output_relu = self.dropoutConv(conv2d_2_output_relu)
        conv2d_3_output = self.conv2d_3(embedding_output_reshaped)
        conv2d_3_output_relu = self.ReLU(conv2d_3_output)
        conv2d_3_output_relu = self.dropoutConv(conv2d_3_output_relu)
        conv2d_4_output = self.conv2d_4(embedding_output_reshaped)
        conv2d_4_output_relu = self.ReLU(conv2d_4_output)
        conv2d_4_output_relu = self.dropoutConv(conv2d_4_output_relu)

        max_0_output = self.maxpool2d_0(conv2d_0_output_relu)
        max_1_output = self.maxpool2d_1(conv2d_1_output_relu)
        max_2_output = self.maxpool2d_2(conv2d_2_output_relu)
        max_3_output = self.maxpool2d_3(conv2d_3_output_relu)
        max_4_output = self.maxpool2d_4(conv2d_4_output_relu)

        linear_input = torch.cat((max_0_output, max_1_output, max_2_output,\
            max_3_output, max_4_output), dim = -1)

        if self.num_filters != 1:
            linear_input = linear_input.reshape(input.shape[0], 1, 1, 5*self.num_channels*self.num_filters).detach().clone()

        linear_0_output = self.linear_0(linear_input)
        linear_0_output_relu = self.ReLU(linear_0_output)
        linear_0_output_relu_dropout = self.dropout(linear_0_output_relu)
        linear_1_output = self.linear_1(linear_0_output_relu_dropout).squeeze()
        
        result = {'pred': linear_1_output}
        # compute loss
        if y is not None:
            loss = self.loss(linear_1_output, y.squeeze())
            result['loss'] = loss

        return result
    
    '''
    def __init__(self, checkpoint_dir='temp\\ppo'):
        
        super(ActorNetwork, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, 'actor_torch_myModel_1.1.pth')
        
        #--------------------------------------------------------
        self.num_actions = 31
        self.num_accepted = 30
        self.num_channels = self.num_actions+self.num_accepted+1
        self.embedding_dim = 50
        self.num_filters = 50

        with open("embedding_matrix.pkl", 'rb') as f:
            embedding_matrix = pickle.load(f)

        vocab_size = embedding_matrix.shape[0]
        print(f"vocab_size: {vocab_size}")
        vector_size = embedding_matrix.shape[1]
 
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_size, padding_idx = 50)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

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
            out_channels = self.num_filters,
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
        self.dropoutConv = nn.Dropout(p=0.4, inplace=False)
        self.linear_0 = nn.Linear(
            in_features = 217*self.num_channels*self.num_filters,
            #in_features = 195,
            out_features = 128)
        # ReLU
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.linear_1 = nn.Linear(
            in_features = 128,
            out_features = self.num_actions)
        #self.softmax = nn.Softmax(3)
        
        self.loss_fn = torch.nn.MSELoss(reduce=True)

        if not torch.cuda.is_available(): # NOT sbagliato
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, input, y: Optional[torch.Tensor] = None):
        #print("\n\n\n\n---FORWARD---\n")
        embedding_output = self.embedding(input)

        embedding_output_reshaped = embedding_output.reshape(input.shape[0], self.embedding_dim, 50, self.num_channels).detach().clone()

        conv2d_0_output = self.conv2d_0(embedding_output_reshaped)
        conv2d_0_output_relu = self.ReLU(conv2d_0_output)
        conv2d_0_output_relu = self.dropoutConv(conv2d_0_output_relu)
        conv2d_1_output = self.conv2d_1(embedding_output_reshaped)
        conv2d_1_output_relu = self.ReLU(conv2d_1_output)
        conv2d_1_output_relu = self.dropoutConv(conv2d_1_output_relu)
        conv2d_2_output = self.conv2d_2(embedding_output_reshaped)
        conv2d_2_output_relu = self.ReLU(conv2d_2_output)
        conv2d_2_output_relu = self.dropoutConv(conv2d_2_output_relu)
        conv2d_3_output = self.conv2d_3(embedding_output_reshaped)
        conv2d_3_output_relu = self.ReLU(conv2d_3_output)
        conv2d_3_output_relu = self.dropoutConv(conv2d_3_output_relu)
        conv2d_4_output = self.conv2d_4(embedding_output_reshaped)
        conv2d_4_output_relu = self.ReLU(conv2d_4_output)
        conv2d_4_output_relu = self.dropoutConv(conv2d_4_output_relu)

        linear_input = torch.cat((conv2d_0_output_relu, conv2d_1_output_relu, conv2d_2_output_relu,\
            conv2d_3_output_relu, conv2d_4_output_relu), dim = 2)

        #if self.num_filters != 1:
        linear_input = linear_input.reshape(input.shape[0], 1, 1, 217*self.num_channels*self.num_filters).detach().clone()

        linear_0_output = self.linear_0(linear_input)
        linear_0_output_relu = self.ReLU(linear_0_output)
        linear_0_output_relu_dropout = self.dropout(linear_0_output_relu)
        linear_1_output = self.linear_1(linear_0_output_relu_dropout).squeeze()
        
        result = {'pred': linear_1_output}
        # compute loss
        if y is not None:
            loss = self.loss(linear_1_output, y)
            result['loss'] = loss

        return result
    '''

    def loss(self, pred, y):
        return self.loss_fn(pred, y)
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



class Agent:
    #def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            #policy_clip=0.2, batch_size=64, n_epochs=10):
    def __init__(self, alpha=0.0003, batch_size=64, n_epochs=10):

        self.n_epochs = n_epochs

        self.actor = ActorNetwork()
        print("PAREMETERS")
        for name, param in self.actor.named_parameters():
            if name == 'embedding.weight':
                param.requires_grad = False
            if param.requires_grad:
                print(name, param.data)

        #self.actor.optimizer = optim.RMSprop(self.actor.parameters())
        self.actor.optimizer = optim.Adam(self.actor.parameters())
        self.memory = Memory(batch_size)
       
    def remember(self, state, output, ground_truth):
        self.memory.store_memory(state, output, ground_truth)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()

    def load_models_path(self, checkpoint_file):
        print('... loading models ...')
        self.actor.load_checkpoint_path("actor_"+checkpoint_file)

    def choose_action(self, observation):
        #print("\n---OBSERVATION---")

        state = torch.tensor(observation).long().to(self.actor.device)

        with torch.no_grad():
            batch_out = self.actor(state.unsqueeze(0))
        output = batch_out['pred']
        dist = torch.softmax(output, dim=0)

        randomNumber = random.randint(0, 39)
        if randomNumber == 0:
            print(f"Action distribution: {dist}")

        dist = Categorical(dist)
        
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()

        return action, output.tolist()

    def learn(self):
        #torch.autograd.set_detect_anomaly(True)
        for _ in range(self.n_epochs):
            state_arr, output_arr, ground_truth_arr, batches = \
                    self.memory.generate_batches()

            #print(self.memory.generate_batches())
            #print(f"BATCHES: {batches}")

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.int64).to(self.actor.device)
                
                ground_truths = torch.tensor(ground_truth_arr[batch]).to(self.actor.device)
                ground_truths = ground_truths.float()
                #outputs.requires_grad = True
                self.actor.optimizer.zero_grad()
                
                #ground_truths.requires_grad = True
                batch_out = self.actor(states, ground_truths)
                #outputs = batch_out['pred']
    
                #mse_loss = nn.MSELoss(reduce=True)
                #actor_loss = mse_loss(outputs, ground_truths)
                actor_loss = batch_out['loss']
                #actor_loss = actor_loss.mean()
                actor_loss.backward()
                '''
                for name, param in self.actor.named_parameters():
                    print(name)
                    print(param.grad)
                '''
                self.actor.optimizer.step()

        print("\n---LOSSES---")
        print(f"actor: {actor_loss}")

        wandb.log({'Train actor_loss': actor_loss})

        self.memory.clear_memory()


