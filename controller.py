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


class State(): 
    
    def __init__(self,x_, y_, theta_): 
        
        if not (x_ is None or y_ is None or theta_ is None ):    #non ricordo perchè avevo messo questo 
            
            self.x = x_ 
            self.y = y_ 
            self.theta = theta_ 
            
        else : 
            
            self.x = 0 
            self.y= 0
            self.theta= 0
            
    def __str__(self): 
        return str(self.x)+","+str(self.y)+","+str(self.theta)
    
    def distance_form_state(self,other): 
        
        distance = np.array([self.x-other.x,self.y-other.y])
        distance_err = distance@distance
        return distance_err
        
        


class Controller():
    
    def __init__(self, goal, start, R_=0.0325,L_=0.1,dT=0.1,v_=1,k_filter=None,only_noise=False,observation_noise=1,process_noise=0.1,arrive_distance=0.3,near_distance=7,check_ac=False,obstacle=None):
    
        self.R= R_ 
        
        self.L = L_, 
        
        self.E= 0   #cumulative error
        
        self.old_e = 0  #previous error 
        
        self.v = v_ 
        
        self.dt = dT 
        
        self.arrive_distance = arrive_distance 
        
        self.goal = goal 
        
        self.current_position = start    ####current position is start 
        
        self.route_x = [] 
        
        self.route_y = []
        
        self.route_theta = [] 
        
        self.observation_noise = observation_noise 
        
        self.process_noise = process_noise 
        
        self.predicted_position_x = 0 
        
        self.predicted_position_y = 0 
        
        self.predicted_position_theta = 0 
        
        self.kalman_filter = k_filter 
        
        self.only_noise = only_noise 
        
        if k_filter is not None:   #I add noise to the estimated position that at this point is the start position
            
            self.estimated_position_x = self.current_position.x + np.random.normal(0,observation_noise)
            
            self.estimated_position_y = self.current_position.y + np.random.normal(0,self.observation_noise)
            
            self.estimated_position_theta = self.current_position.theta + np.random.normal(0,self.observation_noise/10)
            
        else: 
            
            self.estimated_position_x = self.current_position.x 
            
            self.estimated_position_y = self.current_position.y 
            
            self.estimated_position_theta = self.current_position.theta 
            
        self.F = np.eye(3) 
        
        self.Q = np.eye(3)*process_noise**2 
        self.Q[2,2]=(process_noise/10)**2
        
        self.R = np.eye(3)*observation_noise**2 
        self.R[2,2]= (observation_noise/10)**2
        
        self.P = np.eye(3)*observation_noise
        self.P[2,2]= observation_noise/10
    
        #self.innovation_history = []
        
        self.check_ac = check_ac 
        
        self.obstacle = obstacle 
        
        self.near_distance = near_distance 
        
        if self.check_ac: 
            
            self.innovation_history = [] 
            
            self.covariance_inn_hisotry= []
            
            
    def reset_controller(self, goal, start, obstacle= None): 
            
        self.goal = goal 
            
        self.current_position = start 
            
        self.obstacle = obstacle 
            
        self.E = 0 
            
        self.old_e = 0 
            
        self.predicted_position_x = 0 
            
        self.predicted_position_y = 0 
            
        self.predicted_position_theta = 0 
            
        self.route_x = [] 
            
        self.route_y = [] 
            
        self.route_theta = []
            
        if self.kalman_filter is not None: 
        
            self.estimated_position_x = self.current_position.x + np.random.normal(0,self.observation_noise)
        
            self.estimated_position_y = self.current_position.y + np.random.normal(0,self.observation_noise)
        
            self.estimated_position_theta = self.current_position.theta +np.random.normal(0,self.observation_noise/10)
            
        else: 
            
            self.estimated_position_x = self.current_position.x 
            
            self.estimated_position_y = self.current_position.y
            
            self.estimated_position_theta = self.current_position.theta
                
    def iterate_with_gains(self, k, ki, kd, obstacle=False): 
            
        if(obstacle): 
            #if there is an obstacle the robot will move in the tangent of the circumference
            #centered in the obstacle and having radius the distance form the obstacle and the 
            #position of the robot    
            #there is some control next to ensure that the robot exits this circular motion
            dx = -(self.estimated_position_y-self.obstacle.y)    
                                                              
           
            dy = self.estimated_position_x -self.obstacle.x
            
            
   
            
        else :
                
            dx = self.goal.x -self.estimated_position_x 
        
            dy = self.goal.y -self.estimated_position_y
         
        #In the case I have only noise and I'm not using the kalman filter, 
        #I add the observation noise here, because in this case the estimated position is the real position 
        #When I use the kalman filter I already add the observation noise in another step, 
        #this is not when I do the observation, this is where I use the estimated position       
                
        if self.only_noise: 
            
            dx += np.random.normal(0,self.observation_noise)
            dy += np.random.normal(0,self.observation_noise)
          # is None: 
        g_theta = np.arctan2(dy,dx)
                
        
      
            
            
        alpha = g_theta -self.estimated_position_theta
            
        if self.only_noise: 
            alpha = g_theta -self.estimated_position_theta + np.random.normal(0,self.observation_noise/10)
                
        e = np.arctan2(np.sin(alpha), np.cos(alpha))
            
        e_p = e 
        
        e_I = self.E + e 
        
        e_D = e -self.old_e 
        
        self.E = self.E + e
        
        self.old_e = e
        
           
        w = k*e_p + ki*e_I + kd*e_D
         
        return w 
        
        
    def make_prediction(self, w): 
        #prediction with the kalman filter using the estimated postion
        a = self.estimated_position_x +self.v*np.cos(self.estimated_position_theta)*self.dt 
        b = self.estimated_position_y +self.v*np.sin(self.estimated_position_theta)*self.dt 
        
        
        self.predicted_position_x = a 
        self.predicted_position_y = b
        
        theta_dt = w 
        
           
        self.predicted_position_theta = self.estimated_position_theta +theta_dt*self.dt
            
        
    def evaluate_matrices(self): 
        
        self.F[0,2] = -self.v*np.sin(self.estimated_position_theta)*self.dt
        
        self.F[1,2]= self.v*np.cos(self.estimated_position_theta)*self.dt
        
        PP = deepcopy(self.P)
        
        self.P = np.matmul(np.matmul(self.F,PP), np.transpose(self.F)) + self.Q
        
    def simulation_step(self,w): 
        
        x_dt = self.v*np.cos(self.current_position.theta)
        
        y_dt = self.v*np.sin(self.current_position.theta)
        
        theta_dt = w
        
       # self.current_position.theta = (self.fixAngle(
            #self.current_position.theta + self.fixAngle(theta_dt * self.dt))).squeeze(-1)
        
        self.current_position.theta = self.current_position.theta +theta_dt*self.dt
        self.current_position.x = self.current_position.x + x_dt * self.dt
        
        self.current_position.y = self.current_position.y + y_dt * self.dt
        
        if self.kalman_filter is None and not self.only_noise: 
            
            self.estimated_position_x = self.current_position.x 
            self.estimated_position_y = self.current_position.y 
            self.estimated_position_theta = self.current_position.theta 
            
        else: 
            
            self.current_position.x += np.random.normal(0,self.process_noise)
            
            self.current_position.y += np.random.normal(0,self.process_noise)
            
            self.current_position.theta+= np.random.normal(0,self.process_noise/10)
            
            
        if self.only_noise: 
            
            self.estimated_position_x = self.current_position.x 
            self.estimated_position_y = self.current_position.y 
            self.estimated_position_theta = self.current_position.theta 
            
       
            
        return      
        
    def correction(self): 
        
        #here i do the correction and observe the model, 
        #so here is when I add the observation noise in the case of kalman filter 
        
        x = self.current_position.x +np.random.normal(0,self.observation_noise)
        
        y = self.current_position.y+np.random.normal(0,self.observation_noise)
        
        theta = self.current_position.theta+np.random.normal(0,self.observation_noise/10)
        
        observation = np.array([x,y,theta])
        
        innovation = observation - np.array([self.predicted_position_x,self.predicted_position_y,self.predicted_position_theta])
        
        innovation_covariance = self.R +self.P 
        
        if self.check_ac: 
            
            self.innovation_history.append(innovation)
            
            self.covariance_inn_hisotry.append(innovation_covariance)
        
        optimal_gain = np.matmul(self.P,np.linalg.inv(innovation_covariance))
        
        n_x = self.predicted_position_x
        
        n_y = self.predicted_position_y 
        
        n_theta = self.predicted_position_theta 
          
        new_estimation = np.array([n_x,n_y,n_theta])+ np.matmul(optimal_gain,innovation)
        
        PP = deepcopy(self.P)
        
        self.P = np.matmul((np.eye(3) - optimal_gain), PP)
        
        self.estimated_position_x = new_estimation[0]
        
        self.estimated_position_y = new_estimation[1]
        
        self.estimated_position_theta = new_estimation[2]
        
    def isArrived(self):
        #based on the real position
        current_state = np.array([self.current_position.x, self.current_position.y])
        goal_state = np.array([self.goal.x, self.goal.y])
        #goal_state = np.array([self.goal.x, self.goal.y])
        difference = current_state - goal_state

        distance_err = difference @ difference.T
        if distance_err < self.arrive_distance:
            #print("Total number of steps to reach goal : ", self.tot_steps) 
            return True
        else:
            return False 
        
    def nearObstacle(self):
        
        if self.obstacle is None: 
            return False 
        pos_x = self.estimated_position_x 
        pos_y = self.estimated_position_y 
        if(self.only_noise): 
            pos_x += np.random.normal(0,self.observation_noise)
            pos_y += np.random.normal(0,self.observation_noise)
            
        #distance_target = np.array([self.estimatedx,self.current_position.y])
        pos = np.array([pos_x,pos_y])
        goal = np.array([self.goal.x,self.goal.y])
        #distance=np.array([pos_x-self.goal.x, pos_y-self.goal.y])
        distance = pos-goal
        distance_goal = distance@distance.T
        obstacle = np.array([self.obstacle.x,self.obstacle.y])
        #distance = ([pos_x-self.obstacle.x,pos_y -self.obstacle.y])
        distance = pos-obstacle 
        distance_obstacle = distance@distance.T
        if distance_obstacle >=self.near_distance: 
            return False
        if distance_obstacle < distance_goal: 
            return True 
        return False 
        
  
        
    def crush_obstacle(self): 
        #this is obviously based on the real positions 
        if self.obstacle is None: 
            return False 
        
        current_state = np.array([self.current_position.x,self.current_position.y])
        obstacle_state = np.array([self.obstacle.x, self.obstacle.y])
        difference = current_state -obstacle_state 
        distance = difference @ difference.T
        if distance<=self.arrive_distance: 
            return True 
        return False 
        
    
    def relative_pos(self):
        #in order not to go around the obstacle forever in the main algorithm I use the relative position between the obstacle and the robot


        a = self.obstacle.x -self.estimated_position_x   
        b = self.obstacle.y -self.estimated_position_y 
        #when I have noise but not kalman filter I add observ noise here, because in that case in estimated position is saved the real one 
        if self.only_noise: 
            a += np.random.normal(0,self.observation_noise)
            b += np.random.normal(0,self.observation_noise)
        if a>0: 
            a = 1 
        else: 
            a = 0 
            
        if b>0:
            b=1 
        else: 
            b= 0
        return a,b
        
    def track_route(self): 
        
        self.route_x.append(self.current_position.x)
        
        self.route_y.append(self.current_position.y)
        
        self.route_theta.append(self.current_position.theta)
        
    def give_route(self): 
        dt = [self.dt*i for i in range(len(self.route_x))]
        return self.route_x, self.route_y, self.route_theta, dt 
    
    
    def fixAngle(self, angle):
        
        return np.arctan2(np.sin(angle), np.cos(angle))
        
    def check_kalman_filter(self):
    
        return self.innovation_history, self.covariance_inn_hisotry            
            
     
        
        
        
        
        
        
            
            
        
            
            
            
            
            
            
            
    
