'''
This script helps in plotting the gaussian distribution.
'''

# importing libraries
import matplotlib.pyplot as plt

def histogram(x, y, bin_size, label):
    # plotting the histogram with bin size = 20
    plt.hist(x, bins = bin_size, color = "blue");
    plt.title("Gaussian Distribution (random variable X)")
    plt.xlabel(label)
    plt.show()
    plt.hist(y, bins = bin_size, color = "orange");
    plt.title("Gaussian Distribution (random variable Y)")
    plt.xlabel(label)
plt.show()