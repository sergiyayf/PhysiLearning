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
    sen_total_sum = np.sum((sen_simulation[1:] - sen_ground_truth[1:])**2)
    res_total_sum = np.sum((res_simulation[1:] - res_ground_truth[1:])**2)
    return sen_total_sum, res_total_sum


def run_simulation(initial_guess=[1,2,3], initial_condition=[0.99,0.01], exp_data = [1.1, 0.01], iterations=6, treatment_type='NT'):
    #growth_rate_sus, growth_rate_res, treat_sus = initial_guess
    # growth_rate_res, growth_rate_sus, random_death, death_treat_sus,
    # competition, ramp_time, min_treat
    (growth_rate_sus, growth_rate_res, random_death, death_treat_sus, competition,
     ramp_time_up, ramp_time_down, carrying_capacity, min_treat,
     on_treat_threshold, off_treat_threshold) = initial_guess
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
    evaluation.env.env.growth_rate = [growth_rate_sus, growth_rate_res]
    evaluation.env.env.death_rate_treat[0] = death_treat_sus
    evaluation.env.env.death_rate = [random_death, random_death]
    evaluation.env.env.competition[0] = competition
    evaluation.env.env.env_specific_params['ramp_time_up'] = ramp_time_up
    evaluation.env.env.env_specific_params['ramp_time_down'] = ramp_time_down
    evaluation.env.env.capacity = carrying_capacity
    evaluation.env.env.env_specific_params['min_death_during_treat'] = min_treat
    evaluation.env.env.on_treat_threshold = on_treat_threshold
    evaluation.env.env.off_treat_threshold = off_treat_threshold

    #print("Initial observation:", obs)
    sus.append(evaluation.env.env.initial_wt)
    res.append(evaluation.env.env.initial_res)
    obs = (evaluation.env.env.initial_wt, evaluation.env.env.initial_res)
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
        elif treatment_type == 'AT100':
            if exp_data[0][i] + exp_data[1][i] >= exp_data[0][0] + exp_data[1][0]:
            #if obs[0] + obs[1] >= 1.0:
                action = 1
            else:
                action = 0

        obs, reward, term, trunc, info = evaluation.env.step(action)

        if i == 0 and exp_data is not None:
            #evaluation.env.env.state[0] = exp_data[0][1]
            evaluation.env.env.state[1] = exp_data[1][1]
            obs[1] = exp_data[1][1]
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
    data = pd.read_csv(path, sep='\s', names=['name', 'sus', 'res'], engine='python')

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
    base = '/home/saif/Projects/PhysiLearning/data/experimental_data_elias/2025_stem_paper_final_final'
    experiment_replicas = [(1, 'B4'), (1, 'B5'), (1, 'C4'), (1, 'C6'), (1, 'D4'),
                           (2, 'A1'), (2, 'A6'), (2, 'C1'), (2, 'C2'), (2, 'D2'), (2, 'D5'),
                           (3, 'B2'), (3, 'B3'), (3, 'D2'), (3, 'D4'),
                           (4, 'A4'), (4, 'A5'), (4, 'B1'), (4, 'B6'), (4, 'C2'), (4, 'C5'),
                           (5, 'A2'), (5, 'A5'), (5, 'B3'), (5, 'B5'), (5, 'C1'), (5, 'C3'),
                           (5, 'C4'), (5, 'C5'), (5, 'C6'), (5, 'D2'), (5, 'D5')
                           ]
    # experiment_replicas = [(5, 'C5'), (5, 'D5'), (4, 'A5')]
    experiments = ['Plate 1/Results/Results_G418_a2a4_B4_AT100-B4.txt',
                   'Plate 1/Results/Results_G418_a2a4_B5_AT100-B5.txt',
                   'Plate 1/Results/Results_G418_a2a4_C4_Pulse-C4.txt',
                   'Plate 1/Results/Results_G418_a2a4_C6_Pulse-C6.txt',
                   'Plate 1/Results/Results_G418_a2a4_D4_NT-D4.txt',
                   'Plate 2/Results/Results_G418_a2a4_A1_CT-A1.txt',
                   'Plate 2/Results/Results_G418_a2a4_A6_CT-A6.txt',
                   'Plate 2/Results/Results_G418_a2a4_C1_AT100-C1.txt',
                   'Plate 2/Results/Results_G418_a2a4_C2_AT100-C2.txt',
                   'Plate 2/Results/Results_G418_a2a4_D2_Pulse-D2.txt',
                   'Plate 2/Results/Results_G418_a2a4_D5_Pulse-D5.txt',
                   'Plate 3/Results/Results_G418_a2a4_B2_CT-B2.txt',
                   'Plate 3/Results/Results_G418_a2a4_B3_CT-B3.txt',
                   'Plate 3/Results/Results_G418_a2a4_D2_AT100-D2.txt',
                   'Plate 3/Results/Results_G418_a2a4_D4_AT100-D4.txt',
                   'Plate 4/Results/Results_G418_a2a4_A4_Pulse-A4.txt',
                   'Plate 4/Results/Results_G418_a2a4_A5_Pulse-A5.txt',
                   'Plate 4/Results/Results_G418_a2a4_B1_NT-B1.txt',
                   'Plate 4/Results/Results_G418_a2a4_B6_NT-B6.txt',
                   'Plate 4/Results/Results_G418_a2a4_C2_CT-C2.txt',
                   'Plate 4/Results/Results_G418_a2a4_C5_CT-C5.txt',
                   'Plate 5/Results/Results_G418_a2a4_A2_AT100-A2.txt',
                   'Plate 5/Results/Results_G418_a2a4_A5_AT100-A5.txt',
                   'Plate 5/Results/Results_G418_a2a4_B3_Pulse-B3.txt',
                   'Plate 5/Results/Results_G418_a2a4_B5_Pulse-B5.txt',
                   'Plate 5/Results/Results_G418_a2a4_C1_NT-C1.txt',
                   'Plate 5/Results/Results_G418_a2a4_C3_NT-C3.txt',
                   'Plate 5/Results/Results_G418_a2a4_C4_NT-C4.txt',
                   'Plate 5/Results/Results_G418_a2a4_C5_NT-C5.txt',
                   'Plate 5/Results/Results_G418_a2a4_C6_NT-C6.txt',
                   'Plate 5/Results/Results_G418_a2a4_D2_CT-D2.txt',
                   'Plate 5/Results/Results_G418_a2a4_D5_CT-D5.txt'
                   ]

    # combine base and experiments
    experiment_replicas = [(os.path.join(base, exp)) for exp in experiments]
    for path in experiment_replicas:
        sus_exp, res_exp = get_data(path)
        name = path[-14:]
        if 'NT' in name:
            treatment_type = 'NT'
        elif 'CT' in name:
            treatment_type = 'CT'
        elif 'Pulse' in name:
            treatment_type = 'Pulse'
        elif 'AT100' in name:
            treatment_type = 'AT100'
        sus, res = run_simulation(initial_guess, initial_condition=[sus_exp[0], res_exp[0]],
                                  exp_data = [sus_exp, res_exp],
                                  iterations=len(sus_exp), treatment_type=treatment_type)
        sen_total_sum, res_total_sum = error_function(sus, res, sus_exp, res_exp)
        results.append(sen_total_sum + res_total_sum)
    print(f"Results: {np.mean(results)}")
    return np.mean(results)


def fit_simulation():
    # params for sim_time = 28000
    initial_guess = [0.17, 0.417, 0.0068, 0.14, 5.56, 7.32, 2.25, 7.96, 0.014, 1.4, 1.91]  # growth_rate_res, growth_rate_sus, random_death, death_treat_sus,
    # fitted_params = [1.739e-01  6.759e-03  1.616e-01  6.222e+00  7.503e+00
    #                   2.134e+00  7.596e+00  1.572e-02  1.475e+00  1.745e+00]
    # params with res growht   1.70938296e-01 4.17564435e-01 6.80164249e-03 1.39281076e-01
    #  5.56382183e+00 7.31752169e+00 2.25522011e+00 7.96293382e+00
    #  1.43646694e-02 1.39999904e+00 1.91250523e+00
    # competition, ramp_time, min_treat
    bounds = [(0.01, 2.9), (0.01, 2.9), (0.00001, 2.5), (0.001, 2.9), (0.5, 20),
              (0.1, 10000), (0.1, 10000), (1,1000), (0.00001, 1.0),
              (0.0001, 10000), (0.0001, 10000)]  # bounds for growth_rate_res and death_treat_sus

    def callback_func(xk):
        print(xk)

    result = opt.minimize(minimization_function, np.array(initial_guess), method='Nelder-Mead', callback=callback_func,
                          options={'disp': True, 'maxiter': 1000}, bounds=bounds)
    optimized_params = result.x

    print(result)
    print("Optimized parameters:", optimized_params)
    #torch.save(result, 'fit_simulation_new_model.pth')
    return optimized_params

def visualize_results(opt_params):
    base = '/home/saif/Projects/PhysiLearning/data/experimental_data_elias/2025_stem_paper_final_final'
    experiment_replicas = [(1, 'B4'), (1, 'B5'), (1, 'C4'), (1, 'C6'), (1,'D4'),
                           (2, 'A1'), (2, 'A6'), (2, 'C1'), (2, 'C2'), (2, 'D2'), (2, 'D5'),
                           (3, 'B2'), (3, 'B3'), (3, 'D2'), (3, 'D4'),
                           (4, 'A4'), (4, 'A5'), (4, 'B1'), (4, 'B6'), (4, 'C2'), (4, 'C5'),
                           (5, 'A2'), (5, 'A5'), (5, 'B3'), (5, 'B5'), (5, 'C1'), (5, 'C3'),
                           (5, 'C4'), (5, 'C5'), (5, 'C6'), (5, 'D2'), (5, 'D5')
                           ]
    #experiment_replicas = [(5, 'C5'), (5, 'D5'), (4, 'A5')]
    experiments = ['Plate 1/Results/Results_G418_a2a4_B4_AT100-B4.txt',
                   'Plate 1/Results/Results_G418_a2a4_B5_AT100-B5.txt',
                   'Plate 1/Results/Results_G418_a2a4_C4_Pulse-C4.txt',
                   'Plate 1/Results/Results_G418_a2a4_C6_Pulse-C6.txt',
                   'Plate 1/Results/Results_G418_a2a4_D4_NT-D4.txt',
                   'Plate 2/Results/Results_G418_a2a4_A1_CT-A1.txt',
                   'Plate 2/Results/Results_G418_a2a4_A6_CT-A6.txt',
                   # 'Plate 2/Results/Results_G418_a2a4_C1_AT100-C1.txt',
                   # 'Plate 2/Results/Results_G418_a2a4_C2_AT100-C2.txt',
                   # 'Plate 2/Results/Results_G418_a2a4_D2_Pulse-D2.txt',
                   # 'Plate 2/Results/Results_G418_a2a4_D5_Pulse-D5.txt',
                   # 'Plate 3/Results/Results_G418_a2a4_B2_CT-B2.txt',
                   # 'Plate 3/Results/Results_G418_a2a4_B3_CT-B3.txt',
                   # 'Plate 3/Results/Results_G418_a2a4_D2_AT100-D2.txt',
                   # 'Plate 3/Results/Results_G418_a2a4_D4_AT100-D4.txt',
                   # 'Plate 4/Results/Results_G418_a2a4_A4_Pulse-A4.txt',
                   # 'Plate 4/Results/Results_G418_a2a4_A5_Pulse-A5.txt',
                   # 'Plate 4/Results/Results_G418_a2a4_B1_NT-B1.txt',
                   # 'Plate 4/Results/Results_G418_a2a4_B6_NT-B6.txt',
                   # 'Plate 4/Results/Results_G418_a2a4_C2_CT-C2.txt',
                   # 'Plate 4/Results/Results_G418_a2a4_C5_CT-C5.txt',
                   # 'Plate 5/Results/Results_G418_a2a4_A2_AT100-A2.txt',
                   # 'Plate 5/Results/Results_G418_a2a4_A5_AT100-A5.txt',
                   # 'Plate 5/Results/Results_G418_a2a4_B3_Pulse-B3.txt',
                   # 'Plate 5/Results/Results_G418_a2a4_B5_Pulse-B5.txt',
                   # 'Plate 5/Results/Results_G418_a2a4_C1_NT-C1.txt',
                   # 'Plate 5/Results/Results_G418_a2a4_C3_NT-C3.txt',
                   # 'Plate 5/Results/Results_G418_a2a4_C4_NT-C4.txt',
                   # 'Plate 5/Results/Results_G418_a2a4_C5_NT-C5.txt',
                   # 'Plate 5/Results/Results_G418_a2a4_C6_NT-C6.txt',
                   # 'Plate 5/Results/Results_G418_a2a4_D2_CT-D2.txt',
                   # 'Plate 5/Results/Results_G418_a2a4_D5_CT-D5.txt'
                   ]

    # combine base and experiments
    experiment_replicas = [(os.path.join(base, exp)) for exp in experiments]
    for path in experiment_replicas:
        sus_exp, res_exp = get_data(path)
        name = path[-14:]
        fig, ax = plt.subplots(figsize=(200 / 72, 150 / 72), constrained_layout=True)
        if 'NT' in name:
            treatment_type = 'NT'
        elif 'CT' in name:
            treatment_type = 'CT'
        elif 'Pulse' in name:
            treatment_type = 'Pulse'
        elif 'AT100' in name:
            treatment_type = 'AT100'
        sus_sim, res_sim = run_simulation(opt_params, initial_condition=[sus_exp[0], res_exp[0]],
                                            exp_data=[sus_exp, res_exp],
                                  iterations=len(sus_exp), treatment_type=treatment_type)

        ax.plot(sus_exp, label=f'Susceptible Exp ({name})', marker='o', color='blue')
        ax.plot(res_exp, label=f'Resistant Exp ({name})', marker='x', color='orange')
        ax.plot(sus_sim, label=f'Susceptible Sim ({name})', linestyle='--', color='blue')
        ax.plot(res_sim, label=f'Resistant Sim ({name})', linestyle='--', color='orange')

        # title with name
        ax.set_title(f'Well {name}, {treatment_type}')

def get_day_wise_data(plate):
    all_data = pd.DataFrame()
    for i in ['01', '03', '05', '08', '10', '12', '15', '17']:
        filepath = f'/home/saif/Projects/PhysiLearning/data/experimental_data_elias/2025_stem_paper_final_final/Plate {plate}/Results_today/Results_d{i}.txt'
        data = pd.read_csv(filepath, sep='\s', names=['name', 'sus', 'res'], engine='python')
        # get day from the name (last 2 digits, starting with 0 if day <10
        data['plate'] = data['name'].str.extract(r"-(.*?)_d").astype(str)
        data['day'] = int(i)
        all_data = pd.concat([all_data, data], ignore_index=True)

    return all_data

def get_well_data(plate, well):
    data = get_day_wise_data(plate)
    well_data = data[data['plate'] == well]
    # normalize to 1
    #ini_tot = well_data['sus'].iloc[0] + well_data['res'].iloc[0]
    #well_data['sus'] /= ini_tot
    #well_data['res'] /= ini_tot
    return well_data

def get_new_data(plate, well):
    data = get_well_data(plate, well)
    name = data['name'].values
    sus_exp = data['sus'].values
    res_exp = data['res'].values
    # normalize data
    ini_tot = sus_exp[0] + res_exp[0]
    sus_exp = sus_exp / ini_tot
    res_exp = res_exp / ini_tot
    # crop if  sus_exp = 0
    if np.any(sus_exp == 0):
        idx = np.where(sus_exp == 0)[0][0]
        sus_exp = sus_exp[:idx]
        res_exp = res_exp[:idx]
    return name[0], sus_exp, res_exp

def main():
    return fit_simulation()



if __name__ == "__main__":
    opt_params = main()
    visualize_results(opt_params)

    # get_data()
    # run_simulation()
    # minimization_function([0.1, 0.1, 0.2])

    #data = get_new_data(plate=1, well='B4')