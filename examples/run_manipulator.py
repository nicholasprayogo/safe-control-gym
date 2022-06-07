
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
    position_goal = [0.03692, 0, 0.973]
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
    
    policy = "A2C"
    
    if policy == "random":
        # # torques 
        action_space = [50, 100, 200, 400]

        for i in range(50):
            # action_index = env.n_joints- 2
            action_index = 2 
            action_list = np.zeros(env.n_joints)
            action_list[action_index] = random.choice(action_space)    
            obs, reward, done, info = env.step(action_list)
            
            if i%5== 0 :
                print(f"position: {obs[action_index][5]}") # link position
                print(f"angular velocity: {obs[action_index][-1]}") # velocity
                        
            sleep(0.1)
            
    else:
        # model = PPO('MultiInputPolicy', env, verbose=1)
        # model.learn(total_timesteps=10)
        # env = gym.make('CartPole-v1')
        
        #TODO action 0 for other joints except 1 
        env.action_space = spaces.Box(300, 400, (env.n_joints,), dtype=np.float32)
        env.observation_space = spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)

        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=10)
        obs = env.reset()
        
        for i in range(100):
            action, _state = model.predict(obs, deterministic=False)
            print(_state) 
            obs, reward, done, info = env.step(action)
            applied_action = env._action_mapping_torque(action)
            
            print(f"Iteration:{i}")
            print(f"action: {action}")
            print(f"goal: {goal[0]['position'][7]}")
            print(f"state: {[round(ob,3) for ob in obs]}")
            print(f"reward: {reward}\n")
            
            sleep(0.05)

if __name__ == "__main__":
    main()