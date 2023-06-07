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
from utility import *

from agent import *
from copy import deepcopy
from controller import *



def simulation(spatial_state, target, algo=0,sim=1200, obstacle=None,path="slow_model",ob_noise=0.1,p_noise=0.01):

    agent = Agent(3,chkpt_dir =path,device='cpu')
    
    agent.load_actor()   #cambiare e fare solo load agent 
    
    k = abs(agent.actor.mu.weight[0].item())
    ki = abs(agent.actor.mu.weight[1].item())
    kd = abs(agent.actor.mu.weight[2].item())
    
    if algo==0: 
    
        controller = Controller(target,spatial_state,obstacle=obstacle) 
        
    elif algo==1: 
    
        controller = Controller(target,spatial_state,only_noise=True,observation_noise=ob_noise,process_noise=p_noise,obstacle=obstacle) 
        
    else: 
    
        controller = Controller(target,spatial_state,k_filter=3,observation_noise=ob_noise,process_noise=p_noise,check_ac=True,obstacle=obstacle) 
        
    done = False 
    
    flag = True #vedere se riesco a incorporare questa nel controller 
    
    num_steps = 0 
    
    x = []
    
    y = [] 
    
    sign_list = [] 
    string = "target not reached"
    controller.track_route()
   # while not done: 
    for ss in range(sim):
   
    
    
        if flag: 
        
            near = controller.nearObstacle() 
            
            if near: 
            
                a,b = controller.relative_pos() 
                
                sign_list.append([a,b]) 
                
                if a!=sign_list[0][0] and b!=sign_list[0][1]: 
                
                    sign_list = [] 
                    
                    flag = False 
                    
                    near = False 
        if not done:             
            w =  controller.iterate_with_gains(k, ki, kd, obstacle=near)     
        
            if algo==2:
        
                controller.make_prediction(w)  
            
                controller.evaluate_matrices()
            
            controller.simulation_step(w)
        
            if algo==2: 
        
                controller.correction() 
            
            crush = controller.crush_obstacle() 
            
            done = controller.isArrived()
            if done: 
                string = "arrived at episode" +str(ss)        
        
            if crush: 
        
                done = True 
                
                string = "crushed at episode" +str(ss)
               
                
        controller.track_route()
        
    x,y,theta,t = controller.give_route()
    
        
    if algo==2: 
    
        innovation, covariance = controller.check_kalman_filter()
        
        return x,y,theta,t,string,innovation,covariance 
        
    return x,y,theta,t,string,0,0
    
    
             
            
                
        
               
    
    
          
    
        
    
    
    


