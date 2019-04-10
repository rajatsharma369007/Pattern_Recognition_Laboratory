'''
This script helps to plot the decision boundary
'''

# importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from utils import discriminant

# plotting
def scatter_plot(x_point, xvec1, xvec2, w1, w2):
    
    # defining plotting scale
    max_x = max(np.append(xvec1[:, 0], xvec2[:, 0]))
    min_x = min(np.append(xvec1[:, 0], xvec2[:, 0]))
    max_y = max(np.append(xvec1[:, 1], xvec2[:, 1]))
    min_y = min(np.append(xvec1[:, 1], xvec2[:, 1]))
    
    plot_class1 = []
    plot_class2 = []
    
    for i in range((min_x-2) * 10 , ((max_x+2)+1) * 10):
        i = i * 0.1
        for j in range((min_y-2) * 10 , ((max_y+2)+1) * 10):
            j = j * 0.1
            x = np.array([i, j])
            g1, g2 = discriminant.function(x, xvec1, xvec2, w1, w2)
            if g1 - g2 > 0:                 # class 1 condition
                plot_class1.append(x)
            else:                           # class 2 condition
                plot_class2.append(x)
    
    # converting the points to numpy array
    plot_class1 = np.array(plot_class1)
    plot_class2 = np.array(plot_class2)
    
    # scatter plot
    plt.scatter(plot_class1[:, 0], plot_class1[:, 1], c="yellow")
    plt.scatter(plot_class2[:, 0], plot_class2[:, 1], c="orange")
    plt.scatter(x_point[0], x_point[1], c="black", label = "input")
    plt.scatter(xvec1[:, 0], xvec1[:, 1], c="green", label = "class1")
    plt.scatter(xvec2[:, 0], xvec2[:, 1], c="red", label = "class2")
    plt.hlines(y=0, xmin=-1, xmax=8, linestyles='dashed')
    plt.vlines(x=0, ymin=-6, ymax=11, linestyles='dashed');
    plt.title("Decision Boundary")
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.legend(loc='upper right')
    plt.show()
    