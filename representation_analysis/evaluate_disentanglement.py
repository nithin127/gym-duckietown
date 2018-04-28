import argparse
import os
import sys
import numpy as np
from skimage import io
import time

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from representation_analysis.models import VAE

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--num_samples_train', type=int, default=20,
                    help='num samples for training')
parser.add_argument('--num_samples_test', type=int, default=10,
                    help='num samples per testing')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--saved_model', type=str, help='Save file to use')
parser.add_argument('--saved_model_classifier', type=str, help='Save file to use for classifier')
parser.add_argument('--state_size', type=int, default=50,
                    help='Size of latent code (default: 50)')
parser.add_argument('--seed', type=int, default=7691, metavar='S',
                    help='Random seed (default: 7691)')
parser.add_argument('--num_per_sample', type=int, default=100,
                    help='L in Disentangling paper (default: 100)')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size (default: 10)')
parser.add_argument('--num_epochs', type=int, default=1,
                    help='num epochs (default: 5)')
#'--saved_model representation_analysis/saves/beta-vae/beta-vae_8400.ckpt'

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

sys.path.append(os.getcwd())

# Monkey-patch because I trained with a newer version.
# This can be removed once PyTorch 0.4.x is out.
# See https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560
import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

final_results = []

for model in os.listdir('representation_analysis/saves/beta-tc-vae/'):
    args.saved_model = os.path.join('representation_analysis/saves/beta-tc-vae/', model)

    # load vae
    try:
        loaded_state = torch.load(args.saved_model)
        step = loaded_state['step']
        model = loaded_state['model']
        args.state_size = model['encoder_mean.weight'].shape[0]
        beta = loaded_state['beta']
        vae = VAE(z_dim=args.state_size, use_cuda=args.cuda)
        vae.load_state_dict(model)
        if args.cuda:
            vae.cuda()
        print('model found and loaded successfully...')
    except:
        print('problem loading model! Check model file!')
        continue

    print('proceeding to train with beta={0} h={1}. Wish me luck!'.format(beta, args.state_size))

    # Dataset
    class disentanglement_dataset(Dataset):
        def __init__(self, dir, targets_dir):
            self.root_dir = os.path.join(os.getcwd(), dir)
            self.num_groups = 3
            self.num_per_sample = args.num_per_sample         #L in https://arxiv.org/pdf/1802.05983.pdf pg 4
            self.images_dir = os.listdir(self.root_dir)
            self.labels = np.load(targets_dir)


        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            dir_to_fetch_from = os.path.join(self.root_dir, os.listdir(self.root_dir)[idx])
            images = []
            file_list = []
            for file in os.listdir(dir_to_fetch_from):
                file_list.append(os.path.join(dir_to_fetch_from, file))
            images = io.concatenate_images(io.imread_collection(file_list))
            file_io = []
            for img in images:
                file_io.append(transforms.ToTensor()(img).unsqueeze(0))
            images = torch.cat(file_io, 0)
            if not args.cuda:
                images = Variable(images)
            else:
                images = Variable(images.cuda())
            _, mu, _, _ = vae(images)

            diff = torch.abs(mu[:self.num_per_sample]-mu[self.num_per_sample:]).data

            sample = {'diff': diff.mean(0), 'label': self.labels[idx]}
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
    disentanglement_classifier = classifier(num_inputs=args.state_size, num_outputs=3)
    if args.cuda:
        disentanglement_classifier.cuda()

    disentanglement_classifier_optimizer = optim.Adam(disentanglement_classifier.parameters(), lr=0.001)

    factor_dataset = disentanglement_dataset(dir='representation_analysis/train_data/',
                                             targets_dir='representation_analysis/train_targets.npy')

    data_loader = DataLoader(dataset=factor_dataset, batch_size=args.batch_size, shuffle=False)
    x = iter(data_loader)

    if not args.saved_model_classifier:
        step = 0
        try:
            for epoch in range(args.num_epochs):
                #then = time.time()
                while step < args.num_samples_train:
                    #now = time.time()
                    #print('time elapsed:{}'.format(int(now - then)))
                    step += 1
                    sample = next(x)
                    if args.cuda:
                        diff = Variable(sample['diff'].cuda())
                    else:
                        diff = Variable(sample['diff'])
                    label = sample['label'].type(torch.LongTensor)
                    if args.cuda:
                        target = Variable(label.cuda())
                    else:
                        target = Variable(label)
                    output = disentanglement_classifier(diff)
                    loss = F.nll_loss(output, target)
                    loss.backward()

                    disentanglement_classifier_optimizer.step()
                    print('Epoch {} Train Step: [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                        epoch+1, step, args.num_samples_train, 100*(step/args.num_samples_train), loss.data[0]))
                    #then = time.time()
                if step % 100 == 0:
                    # save model
                    state = {
                        'model': disentanglement_classifier.state_dict(),
                    }
                    torch.save(state, 'representation_analysis/saves/Higgens_classifier-b{0}-h{1}.ckpt'
                               .format(beta, args.state_size))
            step = 0
        except KeyboardInterrupt:
            # save model
            state = {
                'model': disentanglement_classifier.state_dict(),
            }
            torch.save(state, 'representation_analysis/saves/Higgens_classifier-b{0}-h{1}.ckpt'
                       .format(beta, args.state_size))

        print('training complete! Now to see how we do ...')

    factor_dataset = disentanglement_dataset(dir='representation_analysis/test_data/',
                                             targets_dir='representation_analysis/test_targets.npy')

    data_loader = DataLoader(dataset=factor_dataset, batch_size=args.batch_size, shuffle=True)
    x = iter(data_loader)

    if args.saved_model_classifier:
        loaded_state = torch.load(args.saved_model_classifier)
        model = loaded_state['model']
        disentanglement_classifier_optimizer = VAE(z_dim=args.state_size, use_cuda=args.cuda)
        disentanglement_classifier_optimizer.load_state_dict(model)

    disentanglement_classifier.eval()
    num_same = 0
    total_count = 0
    step = 0
    while step < args.num_samples_test:
        step += 1
        sample = next(x)
        if args.cuda:
            diff = Variable(sample['diff'].cuda())
        else:
            diff = Variable(sample['diff'])
        label = sample['label'].type(torch.LongTensor)
        if args.cuda:
            target = Variable(label.cuda())
        else:
            target = Variable(label)

        output = disentanglement_classifier(diff)
        num_same += torch.eq(output.max(1)[1], target).sum()
        total_count += output.size()[0]
        print('step {}/{} current Higgens metric: {:.4f}'.format(step, args.num_samples_test, 100 * (num_same.data[0] / total_count)))

    print('Higgens metric: [{}/{} ({:.0f}%)]'.format(num_same.data[0], total_count, 100*(num_same.data[0]/total_count)))
    final_results.append([beta, args.state_size, 100*(num_same.data[0]/total_count)])
    np.save('representation_analysis/results_tcvae', np.array(final_results))
