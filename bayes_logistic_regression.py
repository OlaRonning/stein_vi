import pandas as pd
from utils import *

from sklearn.model_selection import cross_val_score  # may need Skorch
from stein_variational_gradient_descent import SteinVariationalGradientDescent

DEBUG = False
if DEBUG:
    from sklearn.svm import SVC


# TODO: run on GPU

def init_particles(num_particles, num_features, mean=to_tensor(0.), concentration=to_tensor(.1), rate=to_tensor(1.)):
    ''' Prior as defined in https://arxiv.org/pdf/1608.04471.pdf '''
    num_particles = (num_particles,)
    num_features = (num_features,)
    alphas = torch.distributions.gamma.Gamma(concentration=concentration, rate=rate).sample(num_particles).squeeze()
    particles = torch.distributions.normal.Normal(mean, 1 / torch.sqrt(alphas)).sample(num_features).squeeze().T
    bias = torch.log(alphas).unsqueeze(1)
    return torch.cat((particles, bias), dim=1)


def get_data():
    # See poc folder for data-set exploration
    # > import scipy.io
    # > mat_path = 'data/covertype.mat'
    # > data = scipy.io.loadmat()
    # > x = data['covtype'][:, 1:]
    # > y = torch.tensor(data['covtype'][:,0])
    # > y[y == 2] = -1

    csv_path = 'data/covtype.data'

    names = ['Elevation', 'Aspect',
             'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
             'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
             'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'] + \
            [f'Wilderness_Area_{i}' for i in range(4)] + \
            [f'Soil_type_{i}' for i in range(40)] + \
            ['Cover_type']

    df = pd.read_csv(csv_path, names=list(map(str.lower, names)))
    inputs = torch.tensor(df.drop('cover_type', axis=1).values)
    targets = torch.tensor(df['cover_type'].values)
    targets[targets == 2.] = -1  # Reproducing binary classification from paper
    targets[targets != -1] = 1
    return inputs, targets


def target_likelihood(inputs, weights):
    return 1 / (1 + torch.exp(torch.dot(-weights.T, inputs)))


if __name__ == '__main__':
    inputs, targets = get_data()
    # TODO: data split
    n, dim = inputs.shape
    num_particles = 100

    particles = init_particles(num_particles, dim)

    SteinVariationalGradientDescent(6000, 0.05, target_dist=target_likelihood, kernel='rbf')
    particles = SteinVariationalGradientDescent.execute(particles)
