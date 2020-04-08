import matplotlib.pyplot as plt
import seaborn as sns


def plot_system(particles):
    #TODO: add kernel density estimation!
    plt.clf()
    sns.kdeplot(particles.squeeze().numpy())
    # plt.hist(target, bins=50, density=True)
    plt.show()
