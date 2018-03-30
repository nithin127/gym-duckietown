#!/usr/bin/env python3

import time
from functools import reduce
import operator

import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def init_weights(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        nn.init.orthogonal(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self, enc_size):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=8)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=1)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=1)

        self.linear1 = nn.Linear(32 * 9 * 14, 256)
        self.linear2 = nn.Linear(256, enc_size)

    def forward(self, img):
        batch_size = img.size(0)

        x = img

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        conv_out = F.leaky_relu(x)
        conv_out = conv_out.view(batch_size, -1)

        x = F.leaky_relu(self.linear1(conv_out))
        mid = F.leaky_relu(self.linear2(x))

        return mid

class Decoder(nn.Module):
    def __init__(self, enc_size):
        super().__init__()

        self.linear1 = nn.Linear(enc_size, 256)
        self.linear2 = nn.Linear(256, 32 * 9 * 14)

        self.deconv1 = nn.ConvTranspose2d(32, 32, 4, stride=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 4, stride=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, 8, stride=8)

    def forward(self, enc):
        batch_size = enc.size(0)

        x = F.leaky_relu(self.linear1(enc))
        x = F.leaky_relu(self.linear2(x))
        x = x.view(batch_size, 32, 9, 14)

        x = self.deconv1(x)
        x = F.leaky_relu(x)

        x = self.deconv2(x)
        x = F.leaky_relu(x)

        x = self.deconv3(x)
        x = F.leaky_relu(x)

        return x

class Model(nn.Module):
    def __init__(self, enc_size=8):
        super().__init__()

        self.encoder = Encoder(enc_size)
        self.decoder = Decoder(enc_size)

        self.enc_linear = nn.Linear(enc_size, 32)
        self.vel_linear = nn.Linear(2, 32)
        self.reenc_linear = nn.Linear(64, enc_size)

        self.apply(init_weights)

    def forward(self, obs, vels):
        batch_size = obs.size(0)

        enc = self.encoder(obs)
        dec = self.decoder(enc)

        x0 = F.leaky_relu(self.enc_linear(enc))
        x1 = F.leaky_relu(self.vel_linear(vels))
        x2 = torch.cat((x0, x1), dim=1)
        enc2 = F.leaky_relu(self.reenc_linear(x2))

        dec2 = self.decoder(enc2)

        return dec, dec2

    def print_info(self):
        modelSize = 0
        for p in self.parameters():
            pSize = reduce(operator.mul, p.size(), 1)
            modelSize += pSize
        print(str(self))
        print('Total model size: %d' % modelSize)

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))

def save_img(file_name, img):
    from skimage import io

    img = img.squeeze(0)
    img = img.clamp(0, 1)
    img = img.data
    img = img.transpose(0, 2).transpose(0, 1)
    img = np.flip(img, 0)
    img = img * 255
    img = img.astype(np.uint8)

    io.imsave(file_name, img)

def load_img(file_name):
    from skimage import io

    # Drop the alpha channel
    img = io.imread(file_name)
    img = img[:,:,0:3] / 255

    # Flip the image vertically
    img = np.flip(img, 0)

    # Transpose the rows and columns
    img = img.transpose(2, 0, 1)

    return make_var(img)

def gen_data():
    obs = env.reset().copy()
    obs = obs.transpose(2, 0, 1)

    # Generate random velocities
    vels = np.random.uniform(low=0.3, high=1.0, size=(2,))

    obs2, reward, done, info = env.step(vels)
    obs2 = obs2.transpose(2, 0, 1)

    return obs, vels, obs2

def gen_batch(batch_size=2):
    obs = []
    vels = []
    obs2 = []

    for i in range(0, batch_size):
        o, v, o2 = gen_data()
        obs.append(o)
        vels.append(v)
        obs2.append(o2)

    obs = make_var(np.stack(obs))
    vels = make_var(np.stack(vels))
    obs2 = make_var(np.stack(obs2))

    return obs, vels, obs2

def make_var(arr):
    arr = np.ascontiguousarray(arr)
    arr = torch.from_numpy(arr).float()
    arr = Variable(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr

def test_model(model):
    obs, vels, obs2 = gen_batch()
    img0 = obs[0:1]
    vels = make_var(np.array([0.8, 0.8])).unsqueeze(0)

    dec, obs2 = model(img0, vels)

    save_img('seg_img.png', img0)
    save_img('img_dec.png', dec)
    save_img('img_obs2.png', obs2)

    for i in range(0, 180):
        try:
            img = load_img('real_images/img_%03d.png' % i)
            img = img.unsqueeze(0)
            _, out = model(img, vels)
            save_img('real_images/img_%03d_recon.png' % i, out)
        except Exception as e:
            print(e)

def train(model, obs, vels, target):
    # Zero the parameter gradients
    optimizer.zero_grad()

    dec, obs2 = model(obs, vels)

    dec_loss = (obs - dec).norm(2).mean()
    obs2_loss = (target - obs2).norm(2).mean()

    #loss = 4 * dec_loss + 1 * obs2_loss

    loss = dec_loss



    loss.backward()
    optimizer.step()


    return loss.data[0]

if __name__ == "__main__":
    env = SimpleSimEnv()
    env.reset()

    model = Model()
    model.print_info()
    if torch.cuda.is_available():
        model.cuda()

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=1e-3
    )

    avg_loss = 0

    for epoch in range(1, 1000000):
        startTime = time.time()
        obs, vels, obs2 = gen_batch()
        genTime = int(1000 * (time.time() - startTime))

        startTime = time.time()
        loss = train(model, obs, vels, obs2)
        trainTime = int(1000 * (time.time() - startTime))

        avg_loss = avg_loss * 0.995 + loss * 0.005

        print('gen time: %d ms' % genTime)
        print('train time: %d ms' % trainTime)
        print('epoch %d, loss=%.3f' % (epoch, avg_loss))

        if epoch == 100 or epoch % 1000 == 0:
            test_model(model)

        #if epoch % 1000 == 0:
        #    model.save('trained_models/angle_model.pt')
