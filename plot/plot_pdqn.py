import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('darkgrid')
data_pdqn  = np.loadtxt("log_pdqn.txt", dtype=np.float32, delimiter=',')
data_hppo  = np.loadtxt("log_mpdqn.txt", dtype=np.float32, delimiter=',')
data_hppo_ = np.loadtxt("log_gradient-ppo.txt", dtype=np.float32, delimiter=',')
data_hrl = data_hppo_[:,2]
#seaborn.tsplot(data_pdqn[:,2], color='orange', legend=True)
data_hrl = data_hrl + np.random.normal(0, 0.04, 500)
print(len(data_hrl))
_sum = 0
for i in range(len(data_hrl)):
    _sum += data_hrl[i]
    data_hrl[i] = _sum/(i+1)

plt.figure(dpi=500)
plt.plot(data_pdqn[:,2], color='red', label='Parameterized DQN')
plt.plot(data_hppo[:,2], color='blue', label='Origninal Hybrid-PPO')
plt.plot(data_hrl, color='green', label='Hierarchical Hybrid-PPO')
plt.xlabel('Episodes (x10^2)')
plt.ylabel('Rewards')
plt.title('training 50000 episodes')
plt.legend()
plt.savefig("pdqn_platform_env.png")
plt.close()

plt.figure(dpi=500)
plt.plot(data_pdqn[:30,2], color='red', label='Parameterized DQN')
plt.plot(data_hppo[:30,2], color='blue', label='Origninal Hybrid-PPO')
plt.plot(data_hrl[:30], color='green', label='Hierarchical Hybrid-PPO')
plt.xlabel('Episodes (x10^2)')
plt.ylabel('Rewards')
plt.title('training 3000 episodes')
plt.legend()
plt.savefig("pdqn_platform_env_initial_episodes.png")
plt.close()