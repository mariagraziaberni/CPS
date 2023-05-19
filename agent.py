import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import numpy as np
import matplotlib.pyplot as plt 
import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import os 

MAX_ACTION = +np.pi 
MIN_ACTION = -np.pi

class ReplayBuffer(): 

    def __init__(self, max_size, input_shape, n_actions): 
        
        self.mem_size = max_size 
        
        self.mem_cntr = 0 
        
        self.state_memory = np.zeros((self.mem_size, input_shape)) 
        
        self.new_state_memory = np.zeros((self.mem_size, input_shape)) 
        
        self.action_memory = np.zeros((self.mem_size, n_actions)) 
        
        self.reward_memory = np.zeros(self.mem_size) 
        
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
        
    def store_transition(self, state, action, reward, state_, done): 
        
        index = self.mem_cntr%self.mem_size 
        
        self.state_memory[index] = state
        
        self.new_state_memory[index] = state_ 
        
        self.terminal_memory[index] = done 
        
        self.reward_memory[index] = reward 
        
        self.action_memory[index] = action 
        
        self.mem_cntr +=1 
        
    def sample_buffer(self, batch_size): 
    
        max_mem = min(self.mem_cntr, self.mem_size) 
        
        batch = np.random.choice(max_mem, batch_size) 
        
        states = self.state_memory[batch] 
        
        states_ = self.new_state_memory[batch] 
        
        actions = self.action_memory[batch] 
        
        rewards = self.reward_memory[batch] 
        
        dones = self.terminal_memory[batch] 
        
        return states, actions, rewards, states_, dones 
        
        
        
class CriticNetwork(nn.Module): 
        
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, name= None, chkpt_dir = "save_m_2"): 
        
        super(CriticNetwork, self).__init__ ()
        
        self.name = name
        
        if name is not None: 
            
            if not os.path.exists(chkpt_dir): 
                
                os.makedirs(chkpt_dir) 
                
            self.checkpoint_file= os.path.join(chkpt_dir,name +'_ddpg') 
            
        
        self.input_dims = input_dims 
        
        self.fc1_dims = fc1_dims 
        
        self.fc2_dims = fc2_dims 
        
        self.n_actions = n_actions 
        
        self.name = name 
        
        #self.checkpoint_dir = chkpt_dir 
        
        #self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3') 
        
        self.fc1 = nn.Linear(self.input_dims+n_actions, self.fc1_dims) 
        
        self.fc2 = nn.Linear(self.fc1_dims+n_actions, self.fc2_dims) 
        
        self.q1 = nn.Linear(self.fc2_dims,1) #scalar value of the critic (state-action value) 
        
        
        
      
        
    def forward(self, state, action): 
    
        q1_action_value = self.fc1(T.cat([state,action],dim=1))
        
        q1_action_value = F.relu(q1_action_value) 
        
        #q1_action_value = self.fc2(q1_action_value)
        
        q1_action_value = self.fc2(T.cat([q1_action_value,action],dim=1))
        
        q1_action_value = F.relu(q1_action_value) 
        
        q1 = self.q1(q1_action_value) 
        
        return q1 
    
    def init_weights(self): 
        
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        
        f3 = 0.003
        
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)

        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        
        T.nn.init.uniform_(self.q1.weight.data, -f3, f3)
        
        T.nn.init.uniform_(self.q1.bias.data, -f3, f3)
        
    def save_checkpoint(self): 
        
        if self.name is not None:
    
            print("...saving...") 
        
            T.save(self.state_dict(),self.checkpoint_file)
        
        
    def load_checkpoint(self): 
    
        if self.name is not None:
    
            print("..loading...") 
        
            self.load_state_dict(T.load(self.checkpoint_file)) 
            
            
class PosLinear(nn.Module): 
    def __init__(self, in_dim, out_dim): 
        super(PosLinear, self).__init__() 
        self.weight = nn.Parameter(T.randn((in_dim,out_dim)))
        #self.bias = nn.Parameter(T.zeros((out_dim,)))
        
    def forward(self, x): 
        return T.matmul(x,T.abs(self.weight)) #+self.bias 
        
        
class ActorNetwork(nn.Module): 
        
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, name=None, chkpt_dir = "save_m_2"): 
        
        super(ActorNetwork, self).__init__ ()
        
        self.name = name
        
        if name is not None: 
            
            if not os.path.exists(chkpt_dir): 
                
                os.makedirs(chkpt_dir) 
                
            self.checkpoint_file= os.path.join(chkpt_dir,name +'_ddpg') 
        
        self.input_dims = input_dims 
        
       # self.fc1_dims = fc1_dims 
        
       # self.fc2_dims = fc2_dims 
        
        self.n_actions = n_actions 
        
        self.name = name 
        
        #self.checkpoint_dir = chkpt_dir 
        
        #self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3') 
        
        #self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims) 
       # self.fc1 = nn.Linear(self.input_dims, self.fc1_dims) 
        
       # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
        
        self.mu = PosLinear(self.input_dims, self.n_actions)  
        
        #self.mu = nn.Linear(self.fc2_dims, self.n_actions)
  
        
    def forward(self, state): 
    
       # prob = self.fc1(state)
        
       # prob= F.relu(prob) 
        
        
        #prob= self.fc2(prob)
   
        #prob = F.relu(prob) 
      
        
        mu = (self.mu(state))*MAX_ACTION
        
        return mu
    
    def init_weights(self): 
        
       # f1 =  1 / np.sqrt(self.fc1.weight.data.size()[0])
        
       # f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        
       # T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        
       # T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
      #  T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)

      #  T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        
        f3 = 0.006 
        
        T.nn.init.uniform_(self.mu.weight.data, 0, f3)
        
      #  T.nn.init.uniform_(self.mu.bias.data, -0.003,0.003)
    
    def save_checkpoint(self): 
        
        if self.name is not None:
    
            print("...saving...") 
        
            T.save(self.state_dict(),self.checkpoint_file)
        
        
    def load_checkpoint(self): 
    
        if self.name is not None:
    
            print("..loading...") 
        
            self.load_state_dict(T.load(self.checkpoint_file)) 
        
        
        
class Agent: 
    
    def __init__(self,input_dim, critic_lr=0.0005,actor_lr=0.0005, tau=0.005, gamma = 1,target_noise=0.2, update_actor_interval = 2, warmup = 1000,max_size = 1000000, layer1_size= 400, layer2_size= 300, batch_size = 100, noise = 0.2,chkpt_dir = "model"):
    
        self.input_dims = input_dim
        
        self.n_actions = 1
    
        self.gamma = gamma 
    
        self.tau = tau 
    
        self.max_action = MAX_ACTION
    
        self.min_action = MIN_ACTION
    
        self.memory = ReplayBuffer(max_size, self.input_dims, self.n_actions) 
    
        self.batch_size = batch_size 
    
        self.learn_step_cntr = 0 
    
        self.time_step = 0 
    
        self.warmup = warmup 
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') 
        print(self.device)
    
        self.update_actor_iter = update_actor_interval
        #(self, input_dims, fc1_dims, fc2_dims, n_actions, name=None, chkpt_dir = "save_m_2"): 
    
        self.actor = ActorNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "actor",chkpt_dir=chkpt_dir).to(self.device)
    
        self.critic_1 = CriticNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "critic_1",chkpt_dir=chkpt_dir).to(self.device) 
    
    
        self.critic_2 = CriticNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "critic_2",chkpt_dir=chkpt_dir).to(self.device)
        
        self.actor.init_weights()
        
        self.critic_1.init_weights() 
        
        self.critic_2.init_weights
        
        self.target_actor = ActorNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "target_actor",chkpt_dir=chkpt_dir).to(self.device)
    
        self.target_critic_1 = CriticNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "target_critic_1",chkpt_dir=chkpt_dir).to(self.device) 
    
        self.target_critic_2 = CriticNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions , name = "target_critic_2",chkpt_dir=chkpt_dir).to(self.device) 
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr = actor_lr) 
        
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),lr = critic_lr) 
        
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),lr = critic_lr) 
        
        self.target_noise = target_noise 
        
        self.noise = noise 
    
    
        self.update_network_parameters(tau=1) 
        
        self.freeze_target_parameters()
    
    def freeze_target_parameters(self): 
        
        for p in self.target_actor.parameters():
            
            p.requires_grad = False 
        
        for p in self.target_critic_1.parameters(): 
            
            p.requires_grad = False 
            
        for p in self.target_critic_2.parameters(): 
            
            p.requires_grad = False 
        
        
    def choose_action(self, observation, expl_noise,test=False): 
    
        #if self.time_step < self.warmup: 
        if (self.time_step < self.warmup and (not test)): 
        
            mu = T.tensor(np.random.normal(scale = expl_noise, size = (self.n_actions,)))
            
        else: 
            if(self.time_step ==self.warmup):
                print("End of warmup")
           
            self.actor.eval()
        
            state = T.tensor(observation, dtype = T.float).to(self.device)
            
            with T.no_grad():
            
                mu = self.actor.forward(state).to(self.device) 
            
            #mu = mu +T.tensor(np.random.normal(scale = self.noise), dtype = T.float).to(self.device) 
            
            mu = mu +T.tensor(np.random.normal(0, self.max_action*expl_noise,size=self.n_actions), dtype = T.float).to(self.device) 
        
        #we have to climp to make sure that the actions are in the right boundaries, because adding the noise 
        #this can not be true 
        
        mu_prime = T.clamp(mu, self.min_action, self.max_action)
        
        self.time_step +=1 
        
        return mu_prime.cpu().detach().numpy()
        
    
    def store_transition(self, state, action, reward, new_state, done): 
    
        self.memory.store_transition(state, action, reward, new_state, done) 
        
    def train(self): 
    
        if self.memory.mem_cntr < self.batch_size: 
        
            return 
        self.critic_1_optimizer.zero_grad() 
        
        self.critic_2_optimizer.zero_grad() 
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size) 
        
        reward = T.tensor(reward, dtype= T.float).to(self.device)  #è sempre lo stesso device quindi non cambia nulla
        
        done = T.tensor(done).to(self.device) 
        
        state = T.tensor(state, dtype= T.float).to(self.device) 
        
        action = T.tensor(action, dtype= T.float).to(self.device) 
        
        state_ = T.tensor(new_state, dtype= T.float).to(self.device) 
        
        with T.no_grad():
            target_actions = self.target_actor.forward(state_) 
        
        noise = T.clamp(T.randn_like(action)*self.target_noise*self.max_action,self.target_noise*self.min_action,self.target_noise*self.max_action)
        
      
        
        target_actions = target_actions + noise
        
        target_actions = T.clamp(target_actions, self.min_action, self.max_action) 
        
        Q_tc1 = self.target_critic_1.forward(state_,target_actions) 
        
        Q_tc2 = self.target_critic_2.forward(state_,target_actions) 
        
        Q1 = self.critic_1.forward(state,action) 
        
        Q2 = self.critic_2.forward(state,action) 
        
        Q_tc1[done] = 0.0 
        
        Q_tc2[done] = 0.0 
        
        Q_tc1= Q_tc1.view(-1) 
        
        Q_tc2 = Q_tc2.view(-1) 
        
        critic_target_value = T.min(Q_tc1,Q_tc2) 
        
        target = reward +self.gamma*critic_target_value
        
        target = target.view(self.batch_size,1) 
        
        #self.critic_1_optimizer.zero_grad() 
        
        #self.critic_2_optimizer.zero_grad() 
        
        q1_loss = F.mse_loss(Q1,target) 
        
        q2_loss = F.mse_loss(Q2,target) 
        
        critic_loss = q1_loss + q2_loss 
        
        critic_loss.backward() 
        
        self.critic_1_optimizer.step() 
        
        self.critic_2_optimizer.step() 
        
        self.learn_step_cntr +=1 
        
        #update actor 
        
        if self.learn_step_cntr % self.update_actor_iter != 0: 
         
            return 
        self.actor.train() 
            
        self.actor_optimizer.zero_grad() 
        
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))

        actor_loss = -T.mean(actor_q1_loss) 
        
        actor_loss.backward() 
        
        self.actor_optimizer.step() 
        
        self.actor.eval()
        
        self.update_network_parameters() 
        
    def update_network_parameters(self, tau = None): 
    
        if tau is None: 
         
            tau = self.tau
        
        actor_params = self.actor.named_parameters() 
        
        critic_1_params = self.critic_1.named_parameters()
        
        critic_2_params = self.critic_2.named_parameters()
        
        
        target_actor_params = self.target_actor.named_parameters()
        
        target_critic_1_params = self.target_critic_1.named_parameters()
        
        target_critic_2_params = self.target_critic_2.named_parameters()
        

        critic_1 = dict(critic_1_params)
        
        critic_2 = dict(critic_2_params)
        
        actor = dict(actor_params)
        
       # target_critic_dict = dict(target_critic_params)
        target_actor = dict(target_actor_params)
        
        target_critic_1 = dict(target_critic_1_params)
        
        target_critic_2 = dict(target_critic_2_params)
         
        
        
        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone()+ \
                                      (1-tau)*target_critic_1[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        
        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone()+ \
                                      (1-tau)*target_critic_2[name].clone()

        self.target_critic_2.load_state_dict(critic_2)
        
        
        for name in actor:
            actor[name] = tau*actor[name].clone()+ \
                                      (1-tau)*target_actor[name].clone()
                                      
        self.target_actor.load_state_dict(actor)
        
        
    def save_models(self): 
        
        self.actor.save_checkpoint()
        
        self.target_actor.save_checkpoint()
        
        self.critic_1.save_checkpoint()
        
        self.critic_2.save_checkpoint()
        
        self.target_critic_1.save_checkpoint()
        
        self.target_critic_2.save_checkpoint()
        
        
    def load_models(self): 
        
        self.actor.load_checkpoint()
        
        self.target_actor.load_checkpoint()
        
        self.critic_1.load_checkpoint()
        
        self.critic_2.load_checkpoint()
        
        self.target_critic_1.load_checkpoint()
        
        self.target_critic_2.load_checkpoint()
        
        
        
        
   
        

