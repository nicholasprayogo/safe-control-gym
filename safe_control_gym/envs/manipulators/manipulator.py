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
                goal = None, 
                goal_type = None, 
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
        self.COST = kwargs["cost"]
        
        self.goal = goal 
        self.goal_type = goal_type # point, trajectory, etc
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
        #TODO update all not just observed?
        for state_key in self.observed_link_state_keys: 
            state_index = self.state_key_indices_dict[state_key]
            self.link_states[state_key][link_index] = link_state[state_index]
    
    def _get_observation(self):
        
        for link_state_key in self.observed_link_state_keys:
            for link_index in self.observed_link_indices:
                link_state = self._link_get_state(link_index)
                self._link_update_state(link_index, link_state)
                
        # TODO expand obs to multidimension later
        obs = self.link_states[link_state_key][link_index]
        
        # ValueError: Error: Unexpected observation shape (4,) for Box environment, please use (1, 4) or (n_env, 1, 4) for the observation shape.
        # obs = np.expand_dims(obs, axis=0)
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
        
    def _get_reward(self):
        # TODO generalize so that can read dict goal for multiple state keys and links
    
        # example goal:
        # goal = [{
        #     "position": [None, None, 2, None, ...],
        #     "orientation": [None, None, 2, None, ...]
        # }, {
        #     "position": [None, None, 2, None, ...],
        #     "orientation": [None, None, 2, None, ...]
        # }]
        
        # trajectory 
        
        reward = 0 
        
        if self.COST == Cost.RL_REWARD:
            for state_key in self.observed_link_state_keys:
                for link_index in self.observed_link_indices:
                    goal = np.array(self.goal[0][state_key][link_index]) #TODO this is for point tracking only
                    state = np.array(self.link_states[state_key][link_index])
                    # rmse = np.sqrt(np.mean((goal-state)**2))
                    loss = np.sum(abs(goal-state))
                    reward -= loss
        
        return reward 
            
    def step(self, action_list:np.array):
        assert len(action_list) == self.n_joints, "size of action_list not equal to n_joints"
        
        if self.target_space == "joint":
            for joint_index in self.controlled_joint_indices:
                action = action_list[joint_index]
                self._joint_apply_action(joint_index, action)
            
            self._pb_client.stepSimulation()
    
        obs = self._get_observation()
        # reward = random.choice(self.reward_space)
        reward = self._get_reward()
        info = {} 
        done = False 
        
        return obs, reward, done, info 
    
    def reset(self):
        obs = self._get_observation()
        # obs = np.array([0.0, 0.0, 0.0, 0.0]) #TODO change to a reasonable reset point
        return obs
