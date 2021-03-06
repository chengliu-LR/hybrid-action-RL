import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('darkgrid')
data_mpdqn  = np.loadtxt("../log_mpdqn_GoalEnv.txt", dtype=np.float32, delimiter=',')
data_pdqn  = np.loadtxt("../log_pdqn_GoalEnv.txt", dtype=np.float32, delimiter=',')
data_paddpg = np.loadtxt("../log_paddpg_GoalEnv.txt", dtype=np.float32, delimiter=',')
data_qpamdp = np.loadtxt("../log_qpamdp_GoalEnv.txt", dtype=np.float32, delimiter=',')
#seaborn.tsplot(data_pdqn[:,2], color='orange', legend=True)

plt.figure(dpi=500)
plt.plot(data_mpdqn[:,-1], color='coral', label='H-PPO')
plt.plot(data_pdqn[:,-1], color='dodgerblue', label='Hier. H-PPO')
plt.plot(data_paddpg[:,-1], color='slategray', label='DDPG')
plt.plot(data_qpamdp[:,-1], color='lightseagreen', label='P-DQN')
plt.xlabel('Episodes (x10^2)')
plt.ylabel('Probability of Success')
plt.title('Success Probability on Training 20000 Episodes')
plt.legend()
plt.savefig("goal_env_success_rate.png")
plt.close()

plt.figure(dpi=500)
plt.plot(data_mpdqn[:,-2], color='coral', label='H-PPO')
plt.plot(data_pdqn[:,-2], color='dodgerblue', label='Hier. H-PPO')
plt.plot(data_paddpg[:,-2], color='slategray', label='DDPG')
plt.plot(data_qpamdp[:,-2], color='lightseagreen', label='P-DQN')
plt.xlabel('Episodes (x10^2)')
plt.ylabel('Rewards')
plt.title('Averaged Return on Training 20000 Episodes')
plt.legend()
plt.savefig("goal_env_averaged_return.png")
plt.close()