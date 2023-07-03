import matplotlib.pyplot as plt
import numpy as np

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
font2 = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
        }
font1 = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

def track(center, pos, path, episodes):
    heatmap = np.zeros((33,33))
    for i in range(1, int(len(center)/2)):
        x = int(center[i+i]/25)
        y = int(center[i+i+1]/25) 
        # print(x,y)
        heatmap[y, x] += 1
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, cmap='Blues' , interpolation='nearest')
    ax.set_xticklabels(['0', '0','170', '340', '510', '680', '850', '1020'], fontdict= font1)
    ax.set_yticklabels(['0', '0','170', '340', '510', '680', '850', '1020'], fontdict= font1)
    plt.title("Movement agent E"+str(episodes), fontdict = font)
    plt.xlabel("Image size in x",fontdict= font2)
    plt.ylabel("Image size in y",fontdict= font2)
    plt.colorbar(im)
    plt.scatter((0,0), (0,0), c='g', marker='o', s = 20)
    for j in range(0, int(len(pos)/2)):
        x_1 = int(pos[j+j]/25)
        y_1 = int(pos[j+j+1]/25)
        plt.scatter((x_1,x_1), (y_1,y_1), c='r', marker='o', s = 20)
    plt.show()
    # save_track = plt.savefig(path +"track_"+str(episodes)+'.png', dpi =300, format ='png')
    return #save_track