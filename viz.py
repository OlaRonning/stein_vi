import matplotlib.pyplot as plt


def plot_system(target_dist, particles):
    #TODO: add kernel density estimation!
    plt.scatter(particles, target_dist(particles))
    plt.show()
