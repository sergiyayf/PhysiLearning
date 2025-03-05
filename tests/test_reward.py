from physilearning.reward import Reward

def test_rewards():
    rewards = Reward()
    assert rewards.reward_shaping_flag == 0

    rewards = Reward(reward_shaping_flag='ttp')
    reward = rewards.get_reward([0.5, 0.4, 1], 0.5, 2)
    assert reward - 1 < 1e-3

    rewards = Reward(reward_shaping_flag='some_reward')
    try:
        reward = rewards.get_reward([0.5, 0.4, 1], 0.1, 2)
    except ValueError:
        assert True
    else:
        assert False
