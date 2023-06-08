# Project for Cyber Physical Systems Exam 
 
 Control of differential drive robots. 
 
 Description of the files: 
 
 -agent.py : contains the reinforcement learning agent to perform the pid tuning.
 
-pid_tuning_and_performance.ipynb : contains a simplified version of the controller, which only include 
 the behavior reach the goal in absence of any noise. In this notebook the tuning of the pid parameters 
 is performed via reinforcement learning. The parameters are  the weights of a neural network which is daved in the folders "slow_model" and "model_". 
 
 -controller.py :  contains the final version of the controller with the behaviors "reach a goal", "avoid   an obstacle" and "reach a goal while avoiding an obstacle". It also contains the implementation of the kalman filter. 
 
-simulation.py : a functon to perform a single simulation. It takes as argument the initial position of the robot, the target and the obstacle if present, the value of the observation and process noise and the path of the directory where the networks containing the parameters of the pid are saved. The parameter algo has value 0 in absence of noise, value 1 when noise is present but no initiative is taken to deal with it, value 2 for using a kalman filter. 



-Model_Checking.ipynb : verification and falsification using Moonligh, must be in the folder  "some_path_/MoonLight/distribution/python".
                       It's important to execute the code contained in this notebook in a sequential way, without repeating the code more then 1 time without reloading the entire notebook.

-utility.py : utility functions for plotting 

Folders : 

requirements: contains the results of the model checking 

report : contains the report of a project

slow_model : contains the actor neural network with the pid parameters. The pid parameters can also be saved in a file without recovering them from the neural network. 




 
![](./animation.gif)
