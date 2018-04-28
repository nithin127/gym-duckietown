import pickle
import numpy as np
import matplotlib.pyplot as plt

og_1 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_1.p','rb'))
og_36 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_36.p','rb'))
og_5453 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_5453.p','rb'))

b1_1 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-vae_b1_h100_8400_1.p','rb'))
b1_36 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-vae_b1_h100_8400_36.p','rb'))
b1_5453 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-vae_b1_h100_8400_5453.p','rb'))


b5_1 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-vae_b5_h100_1.p','rb'))
b5_36 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-vae_b5_h100_36.p','rb'))
b5_5453 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-vae_b5_h100_5453.p','rb'))

b10_1 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-vae_b10_h100_8400_1.p','rb'))
b10_36 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-vae_b10_h100_8400_36.p','rb'))
b10_5453 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-vae_b10_h100_8400_5453.p','rb'))

bt5_1 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-tcvae_b5_h100_1500_1.p','rb'))
bt5_36 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-tcvae_b5_h100_1500_36.p','rb'))
bt5_5453 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-tcvae_b5_h100_1500_5453.p','rb'))

bt10_1 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-tcvae_b10_h100_2200_1.p','rb'))
bt10_36 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-tcvae_b10_h100_2200_36.p','rb'))
bt10_5453 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-tcvae_b10_h100_2200_5453.p','rb'))


#The plots are only wrt episodes; we can also use #of training updates and #global steps

og_1_eps = [og_1['train_reward_avg'][1][t][2] for t in range(len(og_1['train_reward_avg'][0]))]
og_36_eps = [og_36['train_reward_avg'][1][t][2] for t in range(len(og_36['train_reward_avg'][0]))]
og_5453_eps = [og_5453['train_reward_avg'][1][t][2] for t in range(len(og_5453['train_reward_avg'][0]))]
og_n = min(len(og_1_eps), len(og_36_eps), len(og_5453_eps))

b1_1_eps = [b1_1['train_reward_avg'][1][t][2] for t in range(len(b1_1['train_reward_avg'][0]))]
b1_36_eps = [b1_36['train_reward_avg'][1][t][2] for t in range(len(b1_36['train_reward_avg'][0]))]
b1_5453_eps = [b1_5453['train_reward_avg'][1][t][2] for t in range(len(b1_5453['train_reward_avg'][0]))]
b1_n = min(len(b1_1_eps), len(b1_36_eps), len(b1_5453_eps))


b5_1_eps = [b5_1['train_reward_avg'][1][t][2] for t in range(len(b5_1['train_reward_avg'][0]))]
b5_36_eps = [b5_36['train_reward_avg'][1][t][2] for t in range(len(b5_36['train_reward_avg'][0]))]
b5_5453_eps = [b5_5453['train_reward_avg'][1][t][2] for t in range(len(b5_5453['train_reward_avg'][0]))]
b5_n = min(len(b5_1_eps), len(b5_36_eps), len(b5_5453_eps))

b10_1_eps = [b10_1['train_reward_avg'][1][t][2] for t in range(len(b10_1['train_reward_avg'][0]))]
b10_36_eps = [b10_36['train_reward_avg'][1][t][2] for t in range(len(b10_36['train_reward_avg'][0]))]
b10_5453_eps = [b10_5453['train_reward_avg'][1][t][2] for t in range(len(b10_5453['train_reward_avg'][0]))]
b10_n = min(len(b10_1_eps), len(b10_36_eps), len(b10_5453_eps))

bt5_1_eps = [bt5_1['train_reward_avg'][1][t][2] for t in range(len(bt5_1['train_reward_avg'][0]))]
bt5_36_eps = [bt5_36['train_reward_avg'][1][t][2] for t in range(len(bt5_36['train_reward_avg'][0]))]
bt5_5453_eps = [bt5_5453['train_reward_avg'][1][t][2] for t in range(len(bt5_5453['train_reward_avg'][0]))]
bt5_n = min(len(bt5_1_eps), len(bt5_36_eps), len(bt5_5453_eps))

bt10_1_eps = [bt10_1['train_reward_avg'][1][t][2] for t in range(len(bt10_1['train_reward_avg'][0]))]
bt10_36_eps = [bt10_36['train_reward_avg'][1][t][2] for t in range(len(bt10_36['train_reward_avg'][0]))]
bt10_5453_eps = [bt10_5453['train_reward_avg'][1][t][2] for t in range(len(bt10_5453['train_reward_avg'][0]))]
bt10_n = min(len(bt10_1_eps), len(bt10_36_eps), len(bt10_5453_eps))

for k in list(og_1.keys())[:4]:
	fig = plt.figure()
	og_mean = (np.array(og_1[k][0][:og_n]) + np.array(og_36[k][0][:og_n]) + np.array(og_5453[k][0][:og_n]))/3
	og_var = ((og_mean - np.array(og_1[k][0][:og_n]))**2 + (og_mean - np.array(og_36[k][0][:og_n]))**2 + (og_mean - np.array(og_5453[k][0][:og_n]))**2)/3
	og_up = og_mean + 3*og_var
	og_down = og_mean - 3*og_var
	plt.fill_between(og_1_eps[:og_n], og_up, og_down, color='red', alpha = 0.3)
	plt.plot(og_1_eps[:og_n], og_mean, color='red', linewidth='0.4', label='a2c baseline')

	b1_mean = (np.array(b1_1[k][0][:b1_n]) + np.array(b1_36[k][0][:b1_n]) + np.array(b1_5453[k][0][:b1_n]))/3
	b1_var = ((b1_mean - np.array(b1_1[k][0][:b1_n]))**2 + (b1_mean - np.array(b1_36[k][0][:b1_n]))**2 + (b1_mean - np.array(b1_5453[k][0][:b1_n]))**2)/3
	b1_up = b1_mean + 3*b1_var
	b1_down = b1_mean - 3*b1_var
	plt.fill_between(b1_1_eps[:b1_n], b1_up, b1_down, color='purple', alpha = 0.3)
	plt.plot(b1_1_eps[:b1_n], b1_mean, color='purple', linewidth='0.4', label='vae')


	b5_mean = (np.array(b5_1[k][0][:b5_n]) + np.array(b5_36[k][0][:b5_n]) + np.array(b5_5453[k][0][:b5_n]))/3
	b5_var = ((b5_mean - np.array(b5_1[k][0][:b5_n]))**2 + (b5_mean - np.array(b5_36[k][0][:b5_n]))**2 + (b5_mean - np.array(b5_5453[k][0][:b5_n]))**2)/3
	b5_up = b5_mean + 3*b5_var
	b5_down = b5_mean - 3*b5_var
	plt.fill_between(b5_1_eps[:b5_n], b5_up, b5_down, color='aqua', alpha = 0.3)
	plt.plot(b5_1_eps[:b5_n], b5_mean, color='aqua', linewidth='0.4', label='bvae, b=5')

	b10_mean = (np.array(b10_1[k][0][:b10_n]) + np.array(b10_36[k][0][:b10_n]) + np.array(b10_5453[k][0][:b10_n]))/3
	b10_var = ((b10_mean - np.array(b10_1[k][0][:b10_n]))**2 + (b10_mean - np.array(b10_36[k][0][:b10_n]))**2 + (b10_mean - np.array(b10_5453[k][0][:b10_n]))**2)/3
	b10_up = b10_mean + 3*b10_var
	b10_down = b10_mean - 3*b10_var
	plt.fill_between(b10_1_eps[:b10_n], b10_up, b10_down, color='dodgerblue', alpha = 0.3)
	plt.plot(b10_1_eps[:b10_n], b10_mean, color='dodgerblue', linewidth='0.4', label='bvae, b=10')

	bt5_mean = (np.array(bt5_1[k][0][:bt5_n]) + np.array(bt5_36[k][0][:bt5_n]) + np.array(bt5_5453[k][0][:bt5_n]))/3
	bt5_var = ((bt5_mean - np.array(bt5_1[k][0][:bt5_n]))**2 + (bt5_mean - np.array(bt5_36[k][0][:bt5_n]))**2 + (bt5_mean - np.array(bt5_5453[k][0][:bt5_n]))**2)/3
	bt5_up = bt5_mean + 3*bt5_var
	bt5_down = bt5_mean - 3*bt5_var
	plt.fill_between(bt5_1_eps[:bt5_n], bt5_up, bt5_down, color='lime', alpha = 0.3)
	plt.plot(bt5_1_eps[:bt5_n], bt5_mean, color='lime', linewidth='0.4', label='btcvae, b=5')

	bt10_mean = (np.array(bt10_1[k][0][:bt10_n]) + np.array(bt10_36[k][0][:bt10_n]) + np.array(bt10_5453[k][0][:bt10_n]))/3
	bt10_var = ((bt10_mean - np.array(bt10_1[k][0][:bt10_n]))**2 + (bt10_mean - np.array(bt10_36[k][0][:bt10_n]))**2 + (bt10_mean - np.array(bt10_5453[k][0][:bt10_n]))**2)/3
	bt10_up = bt10_mean + 3*bt10_var
	bt10_down = bt10_mean - 3*bt10_var
	plt.fill_between(bt10_1_eps[:bt10_n], bt10_up, bt10_down, color='green', alpha = 0.3)
	plt.plot(bt10_1_eps[:bt10_n], bt10_mean, color='green', linewidth='0.4', label='btcvae, b=10')

	fig.suptitle(k + ' v/s episodes', fontsize=20)
	plt.xlabel('episodes')
	plt.ylabel(k)
	plt.legend()
	fig.savefig(k+'.jpg')


# Turns out that the test thing we're measuring is useless; let's reevaluate that stuff later. Using enjoy.py