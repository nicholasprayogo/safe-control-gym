
from safe_control_gym.envs.manipulators.manipulator import BaseManipulator
from time import sleep
import numpy as np 
import random 
    
def main():    
    urdf_path = "../safe_control_gym/envs/manipulators/assets/franka_panda/panda.urdf"
    control_mode = "torque"
    target_space = "joint"

    env = BaseManipulator(
        urdf_path,
        control_mode,
        target_space 
    )
    
    # torques 
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
        
if __name__ == "__main__":
    main()