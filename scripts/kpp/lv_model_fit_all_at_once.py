import numpy as np
# from distributed.profile import identifier
#
# import model_backbone as mb
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt
import torch
import pandas as pd
from physilearning.train import Trainer
import yaml
from physilearning.evaluate import Evaluation
from simplified_lv import TumorGrowthModel


def error_function(sen_simulation, res_simulation, sen_ground_truth, res_ground_truth):
    # Compute the absolute difference for S+R
    sen_total_sum = np.sum((sen_simulation - sen_ground_truth)**2)
    res_total_sum = np.sum((res_simulation - res_ground_truth)**2)
    return sen_total_sum, res_total_sum


def run_simulation(initial_guess=[1,2,3], initial_condition=[0.99,0.01], iterations=6, treatment_type='NT'):
    #growth_rate_sus, growth_rate_res, treat_sus = initial_guess
    # growth_rate_res, growth_rate_sus, random_death, death_treat_sus,
    # competition, ramp_time, min_treat
    (growth_rate_sus, random_death, death_treat_sus, competition,
     ramp_time_up, ramp_time_down, carrying_capacity, min_treat) = initial_guess
    sus, res = [], []
    config_file = '/home/saif/Projects/PhysiLearning/config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    train = Trainer(config_file)
    train.env_type = 'LvEnv'
    train.setup_env()
    evaluation = Evaluation(train.env, config_file=config_file)

    fixed = config['eval']['fixed_AT_protocol']
    at_type = 'nt'
    threshold = 0.5

    # set up environment with the parameters
    obs = evaluation.env.reset()
    evaluation.env.env.initial_wt = initial_condition[0]
    evaluation.env.env.initial_res = initial_condition[1]
    evaluation.env.env.state = [evaluation.env.initial_wt, evaluation.env.initial_res, 0.0]
    evaluation.env.env.growth_rate = [growth_rate_sus, growth_rate_sus]
    evaluation.env.env.death_rate_treat[0] = death_treat_sus
    evaluation.env.env.death_rate = [random_death, random_death]
    evaluation.env.env.competition[0] = competition
    evaluation.env.env.env_specific_params['ramp_time_up'] = ramp_time_up
    evaluation.env.env.env_specific_params['ramp_time_down'] = ramp_time_down
    evaluation.env.env.capacity = carrying_capacity
    evaluation.env.env.env_specific_params['min_death_during_treat'] = min_treat


    #print("Initial observation:", obs)
    sus.append(evaluation.env.initial_wt)
    res.append(evaluation.env.initial_res)
    for i in range(iterations-1):
        if treatment_type == 'NT':
            action = 0
        elif treatment_type == 'CT':
            action = 1
        elif treatment_type == 'Pulse':
            if i == 0:
                action = 1
            else:
                action = 0
        obs, reward, term, trunc, info = evaluation.env.step(action)

        #print(f"Step {i+1}, Action: {action}, Observation: {obs}, Reward: {reward}")
        sus.append(obs[0])
        res.append(obs[1])

    return sus, res

def run_simplified_simulation(initial_guess=[1,2,3], initial_condition=[0.99,0.01]):
    growth_rate_sus = initial_guess[0]
    model = TumorGrowthModel(max_time=28000, noise=False, growth_rate_wt=growth_rate_sus)
    sus, res = [], []

    sus.append(initial_condition[0])
    res.append(initial_condition[1])
    for i in range(6):
        model.step(0)
        sus.append(model.state[0])
        res.append(model.state[1])
    return sus, res


def get_data(path = '/home/saif/Projects/PhysiLearning/data/experimental_data_elias/2025_08_CoreSeed_Large/Plate_1/Results/Results_G418_a2a4_D1_NT-D1_seg.txt'):
    data = pd.read_csv(path, sep='\s', names=['name', 'sus', 'res'])

    # get day from the name (last 2 digits, starting with 0 if day <10
    data['day'] = data['name'].str.extract(r'(\d{2})$').astype(int)
    # print(data)
    # normalize to 1
    ini_tot = data['sus'].iloc[0] + data['res'].iloc[0]
    data['sus'] /= ini_tot
    data['res'] /= ini_tot
    return data['sus'].values, data['res'].values


def minimization_function(initial_guess):
    results = []
    base ='/home/saif/Projects/PhysiLearning/data/experimental_data_elias/2025_08_CoreSeed_Large'
    experiment_replicas = [f'{base}/Plate_4/Results/Results_G418_a2a4_B{i}_Pulse-B{i}_seg.txt' for i in [2,3]] + \
                          [f'{base}/Plate_5/Results/Results_G418_a2a4_B{i}_NT-B{i}.txt' for i in [1,6]] + \
                          [f'{base}/Plate_3/Results/Results_G418_a2a4_B{i}_CT-B{i}.txt' for i in [2,5]] + \
                          [f'{base}/Plate_5/Results/Results_G418_a2a4_C{i}_CT-C{i}.txt' for i in [2]] + \
                          [f'{base}/Plate_1/Results/Results_G418_a2a4_C{i}_Pulse-C{i}_seg.txt' for i in [5]] + \
                          [f'{base}/Plate_1/Results/Results_G418_a2a4_D{i}_NT-D{i}.txt' for i in [5]]

    for path in experiment_replicas:
        sus_exp, res_exp = get_data(path)
        #print(path)
        #print(f"Sus exp {sus_exp}, Res exp {res_exp}")
        if 'NT' in path:
            treatment_type = 'NT'
        elif 'CT' in path:
            treatment_type = 'CT'
        elif 'Pulse' in path:
            treatment_type = 'Pulse'
        sus, res = run_simulation(initial_guess, initial_condition=[sus_exp[0], res_exp[0]],
                                  iterations=len(sus_exp), treatment_type=treatment_type)
        sen_total_sum, res_total_sum = error_function(sus, res, sus_exp, res_exp)
        results.append(sen_total_sum + res_total_sum)
    print(f"Results: {np.mean(results)}")
    return np.mean(results)


def fit_simulation():
    # params for sim_time = 28000
    initial_guess = [0.20, 0.010, 0.07, 1.27, 125.0, 16.0, 6.5, 0.014]  # growth_rate_res, growth_rate_sus, random_death, death_treat_sus,
    # fitted_params = [0.09794, 0.01192, 0.05102, 2.09967, 153.43139, 16.00365, 8.24394, 0.01486]
    # competition, ramp_time, min_treat
    bounds = [(0.01, 0.9), (0.00001, 0.5),
              (0.001, 0.9), (0.1, 100), (0.1, 10000), (0.1, 10000.0), (2,1000), (0.00001, 1.0)]  # bounds for growth_rate_res and death_treat_sus

    def callback_func(xk):
        print(xk)

    result = opt.minimize(minimization_function, np.array(initial_guess), method='Nelder-Mead', callback=callback_func,
                          options={'disp': True, 'maxiter': 2000}, bounds=bounds)
    optimized_params = result.x

    print(result)
    print("Optimized parameters:", optimized_params)
    #torch.save(result, 'fit_simulation_new_model.pth')
    return optimized_params

def visualize_results(opt_params):
    base = '/home/saif/Projects/PhysiLearning/data/experimental_data_elias/2025_08_CoreSeed_Large'
    experiment_replicas =  [f'{base}/Plate_1/Results/Results_G418_a2a4_D{i}_NT-D{i}.txt' for i in range(1, 3)] + \
                          [f'{base}/Plate_3/Results/Results_G418_a2a4_B{i}_CT-B{i}.txt' for i in range(2, 4)]  + \
                          [f'{base}/Plate_4/Results/Results_G418_a2a4_B{i}_Pulse-B{i}_seg.txt' for i in range(1, 3)]


    for path in experiment_replicas:
        fig, ax = plt.subplots(figsize=(10, 6))
        sus_exp, res_exp = get_data(path)
        if 'NT' in path:
            treatment_type = 'NT'
        elif 'CT' in path:
            treatment_type = 'CT'
        elif 'Pulse' in path:
            treatment_type = 'Pulse'
        sus_sim, res_sim = run_simulation(opt_params, initial_condition=[sus_exp[0], res_exp[0]],
                                  iterations=len(sus_exp), treatment_type=treatment_type)

        ax.plot(sus_exp, label=f'Susceptible Exp ({os.path.basename(path)})', marker='o', color='blue')
        ax.plot(res_exp, label=f'Resistant Exp ({os.path.basename(path)})', marker='x', color='orange')
        ax.plot(sus_sim, label=f'Susceptible Sim ({os.path.basename(path)})', linestyle='--', color='blue')
        ax.plot(res_sim, label=f'Resistant Sim ({os.path.basename(path)})', linestyle='--', color='orange')


def main():
    return fit_simulation()


if __name__ == "__main__":
    opt_params = main()
    visualize_results(opt_params)
    # get_data()
    # run_simulation()
    # minimization_function([0.1, 0.1, 0.2])