import numpy as np
import pandas as pd
from physilearning.reward import Reward
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


class SimOut:
    """
    A class for handling simulation evaluations
    """
    def __init__(
        self,
        sim_type: str = 'Lv',
        patient_id: int = 1,
        therapy_strategy: str = 'random',

    )-> None:

        self.sim_type = sim_type
        self.patient_id = patient_id
        self.therapy_strategy = therapy_strategy

    def get_df(self, run: int = 0):
        df = pd.read_hdf(f'Evaluations/{self.sim_type}/{self.sim_type}EnvEval_patient_{self.patient_id}_{self.therapy_strategy}.h5',
                         key=f'run_{run}')
        return df

    def get_single_reward(self, run: int = 0, flag: int = 0, custom = False, custom_func = None):
        df = self.get_df(run)
        reward = Reward(reward_shaping_flag=flag, normalization=4000)
        res = 0
        # find where df['Type 0'] + df['Type 1'] > 4000
        try:
            index = df.index[(df['Type 0'] + df['Type 1']) >= 4000].tolist()[0]
        except:
            print("Therapy did not fail")
            index = len(df)

        for i in range(index):
            if custom:
                rew = custom_func(df.iloc[i], df.index[i])
            else:
                rew = reward.get_reward(df.iloc[i], df.index[i])
            res += rew
        return res

    def get_average_reward(self, flag: int = 0, custom = False, func = None):
        """
        flag: 0: reward, 1: Type 0, 2: Type 1, 3: Type 2
        """
        outcome = 0
        for i in range(100):
            res = self.get_single_reward(i, flag, custom=custom, custom_func=func)
            outcome += res
        return outcome/100

    def get_all_rewards(self, flag: int = 0, custom = False, func = None):
        """
        flag: 0: reward, 1: Type 0, 2: Type 1, 3: Type 2
        """
        outcome = []
        for i in range(100):
            res = self.get_single_reward(i, flag, custom=custom, custom_func=func)
            outcome.append(res)
        return outcome

    def plot_trajectory(self, ax = None, run: int = 0):
        df = self.get_df(run)
        ax.plot(df.index, df['Type 0'], label='Type 0')
        ax.plot(df.index, df['Type 1'], label='Type 1')
        ax.plot(df.index, df['Type 0'] + df['Type 1'], label='total')
        ax.legend()
        ax.fill_between(df.index, df['Treatment'] * 4000, df['Treatment'] * 4250, color='orange', label='drug', lw=0)

    def plot_box(self, ax = None):
        data = []
        for i in range(100):
            data.append(self.get_single_reward(i, 0))
        sns.boxplot(data=data, ax=ax)

def add_to_df(simulation, df, flag = 0, custom = False, func = None):
    name = simulation.sim_type + '_' + simulation.therapy_strategy + '_' + str(simulation.patient_id)
    df1 = pd.DataFrame({name: simulation.get_all_rewards(flag, custom=custom, func=func)})
    df = pd.concat([df, df1], axis=1)
    return df

def reward_func(obs, time):
    reward = (4000 - 0.75 * (obs[0] + obs[1])) / 4000 - 0.25 * obs[2]
    if sum(obs[0:2]) < 1.e-3:
        reward = 1
    return reward

df = pd.DataFrame()
for patient in [1, 4, 55, 80, 93]:
    for therapy in ['mtd', 'AT50', 'AT75', 'AT100']:

        simulation = SimOut('Lv', patient, therapy)
        df = add_to_df(simulation, df, flag = 7, custom=True, func=reward_func)
fig2, ax2 = plt.subplots()
sns.boxplot(ax=ax2, data=df)
plt.show()

at_sim = SimOut('Lv', 1, 'AT100')
mtd_sim = SimOut('Lv', 1, 'mtd')
rew = at_sim.get_single_reward(0, 0, custom=True, custom_func=reward_func)
mtd_rew = mtd_sim.get_single_reward(0, 0, custom=True, custom_func=reward_func)

for patient in [1, 4, 55, 80, 93]:
    rewards = []
    for therapy in ['mtd', 'AT50', 'AT75', 'AT100']:
        simulation = SimOut('Lv', patient, therapy)
        rewards.append(simulation.get_single_reward(0, 0, custom=True, custom_func=reward_func))
    # find the best therapy
    best = np.argmax(rewards)
    worst = np.argmin(rewards)
    print(f'Patient {patient} best therapy is {["mtd", "AT50", "AT75", "AT100"][best]}')
    print(f'Patient {patient} worst therapy is {["mtd", "AT50", "AT75", "AT100"][worst]}')