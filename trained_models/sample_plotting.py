import pickle
import matplotlib.pyplot as plt

og = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_1.p','rb'))
b1 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-vae_b1_h100_8400_5453.p','rb'))
b5 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-vae_b5_h100_1.p','rb'))
b10 = pickle.load(open('/tmp/gym/a2c/Duckie-SimpleSim-Discrete-v0_beta-vae_b10_h100_8400_5453.p','rb'))


#og_iter = [og['train_reward_avg'][1][t][0] for t in range(len(og['train_reward_avg'][0]))]
#og_steps = [og['train_reward_avg'][1][t][1] for t in range(len(og['train_reward_avg'][0]))]
og_eps = [og['train_reward_avg'][1][t][2] for t in range(len(og['train_reward_avg'][0]))]

#b1_iter = [b1['train_reward_avg'][1][t][0] for t in range(len(b1['train_reward_avg'][0]))]
#b1_steps = [b1['train_reward_avg'][1][t][1] for t in range(len(b1['train_reward_avg'][0]))]
b1_eps = [b1['train_reward_avg'][1][t][2] for t in range(len(b1['train_reward_avg'][0]))]

#b5_iter = [b5['train_reward_avg'][1][t][0] for t in range(len(b5['train_reward_avg'][0]))]
#b5_steps = [b5['train_reward_avg'][1][t][1] for t in range(len(b5['train_reward_avg'][0]))]
b5_eps = [b5['train_reward_avg'][1][t][2] for t in range(len(b5['train_reward_avg'][0]))]

#b10_iter = [b10['train_reward_avg'][1][t][0] for t in range(len(b10['train_reward_avg'][0]))]
#b10_steps = [b10['train_reward_avg'][1][t][1] for t in range(len(b10['train_reward_avg'][0]))]
b10_eps = [b10['train_reward_avg'][1][t][2] for t in range(len(b10['train_reward_avg'][0]))]

for k in list(og.keys())[:4]:
	fig = plt.figure()
	plt.plot(og_eps, og[k][0], color='red', label='a2c baseline')
	plt.plot(b1_eps, b1[k][0], color='blue', label='beta = 1')
	plt.plot(b5_eps, b5[k][0], color='cyan', label='beta = 5')
	plt.plot(b10_eps, b10[k][0], color='purple', label='beta = 10')
	fig.suptitle(k + ' v/s episodes', fontsize=20)
	plt.xlabel('episodes')
	plt.ylabel(k)
	plt.legend()
	fig.savefig(k+'.jpg')


og_eps_test = [og['test_reward'][1][t][2] for t in range(len(og['test_reward'][0]))]
b1_eps_test = [b1['test_reward'][1][t][2] for t in range(len(b1['test_reward'][0]))]
b5_eps_test = [b5['test_reward'][1][t][2] for t in range(len(b5['test_reward'][0]))]
b10_eps_test = [b10['test_reward'][1][t][2] for t in range(len(b10['test_reward'][0]))]

for k in list(og.keys())[4:6]:
	fig = plt.figure()
	plt.plot(og_eps_test, og[k][0], color='red', label='a2c baseline')
	plt.plot(b1_eps_test, b1[k][0], color='blue', label='beta = 1')
	plt.plot(b5_eps_test, b5[k][0], color='cyan', label='beta = 5')
	plt.plot(b10_eps_test, b10[k][0], color='purple', label='beta = 10')
	fig.suptitle(k + ' v/s episodes', fontsize=20)
	plt.xlabel('episodes')
	plt.ylabel(k)
	plt.legend()
	fig.savefig(k+'.jpg')

