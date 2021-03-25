import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style('darkgrid')
data_mpdqn_1 = np.loadtxt("../logs/log_pdqn_GoalEnv_seed_27.txt", dtype=np.float32, delimiter=',')
data_mpdqn_2 = np.loadtxt("../logs/log_pdqn_GoalEnv_seed_4.txt", dtype=np.float32, delimiter=',')
data_mpdqn_3 = np.loadtxt("../logs/log_pdqn_GoalEnv_seed_28.txt", dtype=np.float32, delimiter=',')
data_mpdqn_4 = np.loadtxt("../logs/log_pdqn_GoalEnv_seed_37.txt", dtype=np.float32, delimiter=',')
data_mpdqn_5 = np.loadtxt("../logs/log_pdqn_GoalEnv_seed_98.txt", dtype=np.float32, delimiter=',')

data_pdqn_1 = np.loadtxt("../logs/log_pdqn_single_GoalEnv_seed_14.txt", dtype=np.float32, delimiter=',')
data_pdqn_2 = np.loadtxt("../logs/log_pdqn_single_GoalEnv_seed_44.txt", dtype=np.float32, delimiter=',')
data_pdqn_3 = np.loadtxt("../logs/log_pdqn_single_GoalEnv_seed_69.txt", dtype=np.float32, delimiter=',')
data_pdqn_4 = np.loadtxt("../logs/log_pdqn_single_GoalEnv_seed_72.txt", dtype=np.float32, delimiter=',')
data_pdqn_5 = np.loadtxt("../logs/log_pdqn_single_GoalEnv_seed_80.txt", dtype=np.float32, delimiter=',')

column = 4

#seaborn.tsplot(data_pdqn[:,2], color='orange', legend=True)
training_data_mpdqn = np.array((data_mpdqn_1[:,column], data_mpdqn_2[:,column], data_mpdqn_3[:,column], data_mpdqn_4[:,column], data_mpdqn_5[:,column]))

training_data_pdqn = np.array((data_pdqn_1[:,column], data_pdqn_2[:,column], data_pdqn_3[:,column], data_pdqn_4[:,column], data_pdqn_5[:,column]))

plt.figure(dpi = 300)
sb.tsplot(training_data_pdqn, color='coral')
sb.tsplot(training_data_mpdqn, color='dodgerblue')
plt.xlabel('Episodes (x10^2)')
plt.ylabel('Probability of Success')
plt.title('Success Probability on Training 20000 Episodes')
plt.legend(labels=['p-dqn', 'mp-dqn'])
plt.savefig("goal_env_success_rate_5_runs.png")
plt.close()