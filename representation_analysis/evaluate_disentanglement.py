import argparse
import os
import sys
import numpy as np
from skimage import io, transform

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from representation_analysis.models import VAE

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--saved_model', type=str, help='Save file to use')
parser.add_argument('--state_size', type=int, default=100,
                    help='Size of latent code (default: 100)')
parser.add_argument('--train_len', type=int, default=1000,
                    help='How long to train for (default: 1000)')
parser.add_argument('--num_per_group', type=int, default=1000,
                    help='How many samples do you have per group (default: 100)')
parser.add_argument('--num_per_sample', type=int, default=100,
                    help='L in Disentangling paper (default: 100)')
parser.add_argument('--batch_size', type=int, default=10,
                    help='Batch size of classifier (default: 10)')

#'--saved_model representation_analysis/saves/beta-vae/beta-vae_1000.ckpt'

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#  Load model from before
try:
    loaded_state = torch.load(args.saved_model)
    step = loaded_state['step']
    model = loaded_state['model']
    vae = VAE(z_dim=args.state_size, use_cuda=args.cuda)
    vae.load_state_dict(model)
    optimizer_states = loaded_state['optimizer']
    fixed_x = loaded_state['fixed_x']
    # save_curve(total_losses, TC_losses)
    parameters = list(vae.parameters())
    if args.cuda:
        vae.cuda()

    print('model found and loaded successfully... resuming training from step {}'.format(step))
except:
    print('problem loading model! Check model file!')
    exit(1)

# Dataset
class disentanglement_dataset(Dataset):
    def __init__(self):
        self.root_dir = os.path.join(os.getcwd(), 'representation_analysis/test_data/')
        self.num_groups = 6
        self.num_per_group = args.num_per_group    # How many images per group
        self.num_per_sample = args.num_per_sample  #L in https://arxiv.org/pdf/1802.05983.pdf pg 4

    def __len__(self):
        return args.train_len

    def __getitem__(self, idx):
        #  pick random directory
        group_to_use = np.random.randint(0, self.num_groups)
        dir_to_go_to = os.listdir(self.root_dir)[group_to_use]
        images = []
        for file in range(self.num_per_sample*2):
            file = os.listdir(os.path.join(self.root_dir, dir_to_go_to, 'trajectories'))[np.random.randint(0, self.num_per_group)]
            img_name = os.path.join(self.root_dir, dir_to_go_to, 'trajectories', file)
            images.append(transforms.ToTensor()(io.imread(img_name)).unsqueeze(0))
        images = torch.cat(images, 0)
        if not args.cuda:
            images = Variable(images)
        else:
            images = Variable(images.cuda())
        _, mu, _, _ = vae(images)

        diff = torch.abs(mu[:self.num_per_sample]-mu[self.num_per_sample:]).data

        sample = {'diff': diff.mean(0), 'label': group_to_use}
        return sample


# classifier model
class classifier(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(classifier, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.linear = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, x):
        out = self.linear(x)
        x = F.log_softmax(out)
        return x

#  build classifier
disentanglement_classifier = classifier(num_inputs=args.state_size, num_outputs=6)
if args.cuda:
    disentanglement_classifier.cuda()

disentanglement_classifier_optimizer = optim.SGD(disentanglement_classifier.parameters(),
                                                 lr=0.001, momentum=0.9)

factor_dataset = disentanglement_dataset()
data_loader = DataLoader(dataset=factor_dataset, batch_size=args.batch_size, shuffle=True)

step = 0
for sample in data_loader:
    step += 1
    diff = Variable(sample['diff'])
    if args.cuda:
        target = Variable(sample['label'].cuda())
    else:
        target = Variable(sample['label'])

    output = disentanglement_classifier(diff)
    loss = F.nll_loss(output, target)
    loss.backward()
    disentanglement_classifier_optimizer.step()
    print('Train Step: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        step, len(data_loader), step/len(data_loader), loss.data[0]))

print('done?')
