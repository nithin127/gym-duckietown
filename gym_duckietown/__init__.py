from gym.envs.registration import register

register(
    id='Duckie-SimpleSim-v0',
    entry_point='gym_duckietown.envs:SimpleSimEnv',
    reward_threshold=900.0,
    kwargs={'tinted': False}
)
register(
    id='Duckie-SimpleSim-Tinted-v0',
    entry_point='gym_duckietown.envs:SimpleSimEnv',
    reward_threshold=900.0,
    kwargs={'tinted': True, 'tint_amount': 0.2}
)
