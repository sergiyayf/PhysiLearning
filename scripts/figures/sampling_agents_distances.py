import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import numpy as np
import os
from physilearning.train import Trainer
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
# load trained model from file

os.chdir('/')
model = PPO.load('../../Training/SavedModels/20250109_2DLV_average_less_1_onehalf_day_1_best_reward.zip')
print('Model successfully loaded')

# setup LV environement to run a sample test
trainer = Trainer('../../config.yaml')
trainer.setup_env()
env = trainer.env
# reset environment


# create a list of models
list_of_models = [PPO.load(f'./Training/SavedModels/20250109_2DLV_average_less_1_onehalf_day_{i}_best_reward.zip') for i in range(1, 11)]
# create np array to store actions
num_agents = 10
actions = np.zeros((900, 10))
# run until truncation or termination, print action

t = 0
for model in list_of_models:
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        for m in list_of_models:
            act, _ = m.predict(obs)
            actions[t, list_of_models.index(m)] = act
        obs, reward, done, info = env.step(action)
        t += 1
        print(action)

decision_matrix = np.array(actions)
agent_similarity = 1 - pairwise_distances(decision_matrix.T, metric="hamming")

# Step 4: Perform Clustering (Hierarchical and K-Means)
linkage_matrix = linkage(squareform(1 - agent_similarity), method="ward")
kmeans = KMeans(n_clusters=3, random_state=42).fit(agent_similarity)

# Step 5: Visualization
plt.figure(figsize=(10, 5))
sns.heatmap(agent_similarity, annot=True, cmap="coolwarm", xticklabels=range(1,num_agents+1), yticklabels=range(1,num_agents+1))
plt.title("Agent Similarity Heatmap")
plt.show()