import os
import random
import numpy as np 
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from safe_control_gym.envs.benchmark_env import BenchmarkEnv

class BaseManipulator(BenchmarkEnv):
    NAME = "manipulator"
    
    def __init__(self, 
                urdf_path, 
                control_mode,
                target_space 
                ):
        
        self._pb_client = BulletClient(connection_mode=p.GUI)
        self.urdf_path = urdf_path 
        self.robot = p.loadURDF(urdf_path)
        self.n_joints = self._pb_client.getNumJoints(self.robot) 
        self.link_states = [[] for link in range(self.n_joints)]
        
        control_mode_dict = {
            "torque": p.TORQUE_CONTROL,
            "position": p.POSITION_CONTROL,
            "velocity": p.VELOCITY_CONTROL
        }
        
        self.control_mode = control_mode_dict[control_mode]
        self.target_space = target_space # ["end_effector", "joints"] 
        self.reward_space = [5, 10, 15] # TODO properly define 
        
    def step(self, action_list):
        assert len(action_list) == self.n_joints, "size of action_list not equal to n_joints"
        
        if self.target_space == "joint":
            for action_index, action in enumerate(action_list):
                link_index = action_index
                if self.control_mode == p.TORQUE_CONTROL:
                    self._pb_client.setJointMotorControl2(self.robot, jointIndex=action_index,
                            controlMode= self.control_mode, force=action
                    )
                elif self.control_mode == p.POSITION_CONTROL:
                    self._pb_client.setJointMotorControl2(self.robot, jointIndex=action_index,
                            controlMode= self.control_mode, targetPosition = action
                    )
                
                self.link_states[link_index] = self._pb_client.getLinkState(self.robot,
                                                        linkIndex=link_index,
                                                        computeLinkVelocity=True)
            self._pb_client.stepSimulation()
            
            ## single action
            # action_index = 2
            # action = action_list[action_index]
            # link_index = action_index
            # self._pb_client.setJointMotorControl2(self.robot, jointIndex=action_index,
            #                 controlMode= self.control_mode, targetPosition = action
            #         )
            # self.link_states[link_index] = self._pb_client.getLinkState(self.robot,
            #                                             linkIndex=link_index,
            #                                             computeLinkVelocity=True)
            # self._pb_client.stepSimulation()
            # print("step")
        
        obs = self.link_states
        reward = random.choice(self.reward_space)
        info = {} 
        done = False 
        
        return obs, reward, done, info 
        
    # from manipulator_learning.sim.envs.configs.panda_default import CONFIG
    # from manipulator_learning.sim.robots.manipulator import Manipulator
    
    # make test scripts in separate folder 
    
    # torque and velocity 
    # position : how to know the bounds.  
    # single joint -> constraints
    # multiple joints -> constraints
    
    # fix all the other joints, only move 1 joint 
    
    # control_mode = ["torque", "position", "velocity"]
    
    
    # if end_effector -> apply inverse kinematics to translate to joint control 
    
    # example from manipulator_wrapper
    
    # action[0], action[1], action[2]
    # :param t_command: Translational command.
    # :param r_command: Rotational command. 3 floats for acc or vel command, 4 floats(xyzw quat) for pos.
    # :param g_command: Gripper command.
    # :param ref_frame: Reference frame for action. Should be t or b.
    # :return:
    
    # # these aboe assume position control.
    # # TODO should we make action variable generalize to all control modes 
    # # or diff dimension each time 
    
    # action = {
    #     "torque": [], 
    #     "position": [],
    #     "velocity": [] 
    # }
    
    # _pb_client=p.connect(p.GUI)
        # rc = CONFIG["robot_config"] 
        # # urdf_path = "assets/franka_panda/panda.urdf"
        # # urdf_path = rc["urdf_root"]
        # ee_link_index = rc['ee_link_index']
        # tool_link_index = rc['tool_link_index']
        # gripper_indices = rc['gripper_indices']
        # arm_indices = rc['arm_indices']
        # control_method = "p"
        # gripper_control_method = "p"
        
        # self.manipulator = Manipulator(self._pb_client,
    #             urdf_path,
    #             ee_link_index,
    #             tool_link_index,
    #             control_method,
    #             gripper_control_method,
    #             gripper_indices = gripper_indices,
    #             arm_indices = arm_indices,
    #             # base_constraint = base_constraint
    #             )
