import os
import random
import numpy as np 
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from safe_control_gym.envs.benchmark_env import BenchmarkEnv
from safe_control_gym.envs.benchmark_env import Cost, Task
from gym import spaces 

class BaseManipulator(BenchmarkEnv):
    NAME = "manipulator"
    
    def __init__(self, 
                urdf_path, 
                control_mode,
                target_space, 
                controlled_joint_indices = None, 
                observed_link_indices = None, 
                observed_link_state_keys = None,
                **kwargs 
                ):
        
        self._pb_client = BulletClient(connection_mode=p.GUI)
        self.urdf_path = urdf_path 
        self.robot = p.loadURDF(urdf_path)
        self.n_joints = self._pb_client.getNumJoints(self.robot) 
        self.n_links = self.n_joints
        # self.link_states = [[] for link in range(self.n_joints)]
        
        # if controlled_joint_indices:
        #     # only 1 joint
        #     link_state_placeholder = [[]]
        # else:
        link_state_placeholder = [[] for link in range(self.n_links)]
            
        # this might suite spaces.Dict more 
        self.link_states = {
            "position": link_state_placeholder,
            "orientation": link_state_placeholder,
            "linear_vel": link_state_placeholder,
            "angular_vel": link_state_placeholder
        }
        
        # based on getLinkStates
        # only take world frame. perhaps need base and local frame too?
        # from https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.3v8gjd1epcrt
        self.state_key_indices_dict = {
            "position": 4, 
            "orientation": 5,
            "linear_vel": 6,
            "angular_vel": 7
        }
        
        control_mode_dict = {
            "torque": p.TORQUE_CONTROL,
            "position": p.POSITION_CONTROL,
            "velocity": p.VELOCITY_CONTROL
        }
    
        self.control_mode = control_mode_dict[control_mode]
        self.target_space = target_space # ["end_effector", "joints"] 
        self.reward_space = [5, 10, 15] # TODO properly define 
        
        self.controlled_joint_indices = controlled_joint_indices
        self.observed_link_indices = observed_link_indices # TODO extend these to list
        self.observed_link_state_keys = observed_link_state_keys # which link state to keep track
        
        if not self.observed_link_indices:
            self.observed_link_indices = range(self.n_links)
        
        if not self.controlled_joint_indices:
            self.observed_link_indices = range(self.n_joints)
            
        #TODO use benchmark_env later 
        self.cost = kwargs["cost"]
        #TODO temporary solution
        # self.action_space = action_space 
        # self.observation_space = observation_space 
        # super().__init__(**kwargs)
        
    #TODO wanna make this as flexible? use diff action and obs space?
    # def _set_action_space(self, ):
    #     self.action_space = spaces.Box(-500, 500, (self.n_joints,), dtype=np.float32)
        
    # def _set_observation_space(self):
    #     self.observation_space = spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
    
    def _joint_apply_action(self, joint_index, action):
        if self.control_mode == p.TORQUE_CONTROL:
            self._pb_client.setJointMotorControl2(self.robot, jointIndex=joint_index,
                    controlMode= self.control_mode, force=action
            )
        elif self.control_mode == p.POSITION_CONTROL:
            self._pb_client.setJointMotorControl2(self.robot, jointIndex=joint_index,
                    controlMode= self.control_mode, targetPosition = action
            )
        
    def _link_get_state(self, link_index):
        link_state =  self._pb_client.getLinkState(
            self.robot,
            linkIndex=link_index,
            computeLinkVelocity=True
            )
        return link_state 
    
    def _link_update_state(self, link_index, link_state):
        for state_key, state_index in self.state_key_indices_dict.items(): 
            self.link_states[state_key][link_index] = link_state[state_index]
    
    def _get_observation(self):
        for link_state_key in self.observed_link_state_keys:
            for link_index in self.observed_link_indices:
                link_state = self._link_get_state(link_index)
                self._link_update_state(link_index, link_state)
                obs = self.link_states[link_state_key][link_index]
                
                #TODO observation space might vary depending on size of observed_link_indices observed_link_state_key
                return obs 
        
        # if self.observed_link_indices:
        #     # TODO might want to distinguish which joint to control and which joint/end-effector position to monitor
        #     link_state = self._link_get_state(self.observed_link_indices)
        #     self._link_update_state(link_index, link_state)
            
        #     obs = self.link_states[self.observed_link_state_key][self.observed_link_indices]
        #     return obs 
        
        # else: 
        #     for link_index in range(self.n_links):
        #         link_state = self._link_get_state(link_index)
        #         self._link_update_state(link_index, link_state)
                
        #     obs = self.link_states[self.observed_link_state_key]
        #     return obs 
        
    # def _get_reward(self):
    #     if self.COST == Cost.RL_REWARD:
    #         state = 
            
    def step(self, action_list:np.array):
        assert len(action_list) == self.n_joints, "size of action_list not equal to n_joints"
        
        if self.target_space == "joint":
            # if self.controlled_joint_indices:
            #     action = action_list[self.controlled_joint_indices]
            #     self._joint_apply_action(self.controlled_joint_indices, action)
            # else: 
            #     for joint_index, action in enumerate(action_list):
            #         self._joint_apply_action(joint_index, action)

            for joint_index in self.controlled_joint_indices:
                action = action_list[joint_index]
                self._joint_apply_action(joint_index, action)
            
            self._pb_client.stepSimulation()
    
        # # complete obs 
        obs = self._get_observation()
        
        # only for 1 joint, 1 state
        # obs = self.link_states["position"][6]

        reward = random.choice(self.reward_space)
        info = {} 
        done = False 
        
        return obs, reward, done, info 
    
    def reset(self):
        obs = np.array([0.0, 0.0, 0.0, 0.0]) #TODO change to a reasonable reset point
        return obs
