from physilearning.reward import Reward

def test_rewards():
    rewards = Reward()
    assert rewards.reward_shaping_flag == 0

    rewards = Reward(reward_shaping_flag=1)
    reward = rewards.get_reward([0.5, 0.4], 0.5)
    assert reward-0.5 < 1e-3

    rewards = Reward(reward_shaping_flag=2)
    reward = rewards.get_reward([0.5, 0.4], 0.1)
    assert reward-0.6 < 1e-3

    rewards = Reward(reward_shaping_flag=3)
    reward = rewards.get_reward([0.5, 0.4], 0.1)
    assert reward-0.1 < 1e-3

    rewards = Reward(reward_shaping_flag=4)
    reward = rewards.get_reward([0.5, 0.4, 0], 0.1)
    assert reward-1 < 1e-3

    rewards = Reward(reward_shaping_flag=4)
    reward = rewards.get_reward([0.5, 0.4, 1], 0.1)
    assert reward-0 < 1e-3