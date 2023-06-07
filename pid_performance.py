import numpy as np 

#in this case the reference signals change a bit in the time, so some of these properties are defined in a 
#different way compared to the case where the reference signals is constant 


def overshoot(x,ref): 
    
    if x[0] < ref[0]: 
        
        max_value = np.max(x -ref)
        
        if max_value < 0: 
            
            return 0 
        
        return round(abs(max_value),5)
    
    max_value = np.min(x -ref)
    
    if max_value > 0: 
        
        return 0
    
    return round(abs(max_value),5)
    

def rise_time(x,ref,t): 
    
    diff = x -ref 
    
    sign_ = np.sign(diff)
    
    index = np.min(np.where(sign_!=sign_[0]))
    
    return t[index]
    
    

def settling_time(x,ref,t,perc=1): 
    """
    time required to reach and stay within a specific tolerance 
    band around the steady state
    """
    err = (np.pi*perc)/100
    
    diff = abs(x -ref) 
    
    arr = diff<err 
    
    index = np.max(np.where(arr==False))+1 
    
    if index >= len(x): 
        return t[-1] 
    
    return t[index]
    
        
def steady_state_error(x, ref):
    
    error = round(np.abs(x[-1] - ref[-1]), 5)
    return error    

    
    
    
