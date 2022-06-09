
from safe_control_gym.envs.manipulators.manipulator import BaseManipulator
from safe_control_gym.envs.benchmark_env import Cost, Task
from time import sleep
import numpy as np 
import random 
from stable_baselines3 import A2C, PPO, DDPG
from gym import spaces 

def main():    
    urdf_path = "../safe_control_gym/envs/manipulators/assets/franka_panda/panda.urdf"
    control_mode = "torque"
    target_space = "joint"

    controlled_joint_indices = [6]
    observed_link_indices = [7]
    observed_link_state_keys = ["position"]
    goal = [{
        "position": [None for i in range(13)]
    }]

    # orientation_goal = [0.603, 0.3687, -0.3697, 0.6026]
    # position_goal = [0.03692, 0, 0.973]
    position_goal = [-0.127, 0.0, 1.012]
    # position_goal = [0.0, 0, 1]
    goal[0]["position"][observed_link_indices[0]] = position_goal
    goal_type = "point"

    env = BaseManipulator(
        urdf_path,
        control_mode,
        target_space,
        controlled_joint_indices = controlled_joint_indices,
        observed_link_indices = observed_link_indices, 
        observed_link_state_keys = observed_link_state_keys,
        goal = goal,
        goal_type = goal_type,
        cost = Cost.RL_REWARD
    )
    
    dim_dict = {
        "position":3 + 3,
        "orientation" : 4+4
    }
    env.action_space = spaces.Box(-4.5, 4.5, (len(env.controlled_joint_indices),), dtype=np.float32)

    # TODO expand to multiple state keys 
    env.observation_space = spaces.Box(-np.inf, np.inf, (dim_dict[observed_link_state_keys[0]], ), dtype=np.float32)

    ppo_model = PPO('MlpPolicy', env, verbose=1)
    ppo_model.learn(total_timesteps=10000)
    
    obs = env.reset()
    for i in range(500):
        action, _state = ppo_model.predict(obs, deterministic=False)
        # print(_state)
        obs, reward, done, info = env.step(action)
        
        if i%5==0:
            print(f"Iteration:{i}")
            print(f"action: {action}")
            print(f"goal: {goal[0]['position'][7]}")
            print(f"state: {[round(ob,3) for ob in obs]}")
            print(f"reward: {reward}\n")
        
        if done: 
            print("done")
            break
        
        sleep(0.05)

if __name__ == "__main__":
    main()