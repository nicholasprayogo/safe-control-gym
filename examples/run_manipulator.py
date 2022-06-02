
from safe_control_gym.envs.manipulators.manipulator import BaseManipulator
from time import sleep
import numpy as np 
import random 
from stable_baselines3 import A2C
from gym import spaces 

def main():    
    urdf_path = "../safe_control_gym/envs/manipulators/assets/franka_panda/panda.urdf"
    control_mode = "torque"
    target_space = "joint"

    env = BaseManipulator(
        urdf_path,
        control_mode,
        target_space 
    )
    
    policy = "ppo"
    
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
            
    elif policy == "A2C":
        # model = PPO('MultiInputPolicy', env, verbose=1)
        # model.learn(total_timesteps=10)
        # env = gym.make('CartPole-v1')
        
        #TODO action 0 for other joints except 1 
        env.action_space = spaces.Box(300, 400, (env.n_joints,), dtype=np.float32)
        env.observation_space = spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)

        model = A2C('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=10)

        obs = env.reset()
        for i in range(10):
            action, _state = model.predict(obs, deterministic=True)
            print(action)
            obs, reward, done, info = env.step(action)

if __name__ == "__main__":
    main()