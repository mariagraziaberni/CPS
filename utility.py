import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_average_reward(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    

def func(num, dataSet, line,redDots):
        # NOTE: there is no .set_data() for 3 dim data...
    line.set_data(dataSet[0:2, :num])
    #line.set_2d_properties(dataSet[2, :num])
    redDots.set_data(dataSet[0:2, num-1])
   # redDots.set_2d_properties(dataSet[2, num-1])
   
   
def animation_plot(x,y,tx,ty,ox=None,oy=None,file_name="animation.gif"):

    dataSet= np.array([x,y])
    
    numDataPoints=len(x) 
    if (ox is None or oy is None):
         minx = min(tx,min(x))-5
         miny = min(ty,min(y))-5
         maxx = max(tx,max(x))+5 
         maxy = max(ty,max(y))+5 
    else: 
     
    
        minx = min(tx,ox,min(x))-5
        miny = min(ty,oy,min(y))-5
        maxx = max(tx,ox,max(x))+5 
        maxy = max(ty,oy,max(y))+5 
    
    fig, ax = plt.subplots()
    
    ax.set_xlim(minx,maxx)
    
    ax.set_ylim(miny,maxy)
    
    ax.set_aspect('equal')
    
    ax.grid(True)
    
    line = ax.plot(dataSet[0], dataSet[1], '-', lw=2)[0] # For line plot
    
    redDots = plt.plot(dataSet[0], dataSet[1],  markerfacecolor='red', markeredgecolor='k', marker = "o", markersize=10, label='current position', linestyle='None')[0]
    
    ax.set_xlabel('X')
    
    ax.set_ylabel('y')
    
    ax.set_title("Trajectory") 
    if ox is not None:
        ax.plot(ox,oy,markerfacecolor='red', markeredgecolor='k', marker = "o", markersize=10, label='obstacle', linestyle='None') 
    
    ax.plot(tx,ty,markerfacecolor='green', markeredgecolor='k', marker = "o", markersize=10, label='target', linestyle='None')
    
   # ax.legend(fontsize=15, bbox_to_anchor=(0.74, 0.3, 0.5, 0.5))
    
    ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet,line,redDots), interval=50, blit=False)
    
    ani.save(file_name)
    
    plt.close()
    
    
    
    
    
