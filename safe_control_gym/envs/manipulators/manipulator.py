from email import policy
import os
import random
import numpy as np 
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from safe_control_gym.envs.benchmark_env import BenchmarkEnv
from safe_control_gym.envs.benchmark_env import Cost, Task
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel
from time import sleep
from gym import spaces 
import casadi as cs
#TODO _generate_point_goal on BenchmarkEnv

class BaseManipulator(BenchmarkEnv):
    NAME = "manipulator"
    TASK_INFO = {
        "stabilization_goal": [0, 1],
        "stabilization_goal_tolerance": 0.05,
        "trajectory_type": "circle",
        "num_cycles": 1,
        "trajectory_plane": "xz",
        "trajectory_position_offset": [0.5, 0],
        "trajectory_scale": -0.5
    }
    
    def __init__(self, 
                urdf_path, 
                controlled_variable,
                control_method,
                target_space, 
                controlled_joint_indices, 
                observed_link_indices, 
                observed_link_state_keys,
                goal,
                goal_type, 
                dimensions,
                tolerance, 
                connection = "GUI",
                **kwargs 
                ):
        
        if connection.lower()=="gui":
            connection_mode = p.GUI
        
        else:
            connection_mode = p.DIRECT
        # .disconnect(physicsClientId=self._client)
        
        # self.PYB_CLIENT = -1
        
        # if self.GUI:
        #     self.PYB_CLIENT = p.connect(p.GUI)
        # else:
        #     self.PYB_CLIENT = p.connect(p.DIRECT)
            
        self._pb_client = BulletClient(connection_mode=connection_mode)
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
        # self.state_key_indices_dict = {
        #     "position": 4, 
        #     "orientation": 5,
        #     "linear_vel": 6,
        #     "angular_vel": 7
        # }
        
        self.state_key_indices_dict = {
            "position": 0, 
            "orientation": 1,
            "linear_vel": 6,
            "angular_vel": 7
        }
        
        controlled_variable_dict = {
            "torque": p.TORQUE_CONTROL,
            "position": p.POSITION_CONTROL,
            "velocity": p.VELOCITY_CONTROL
        }
    
        self.controlled_variable = controlled_variable_dict[controlled_variable]
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
        
        self.MIN_TORQUE_POLICY = 2.5
        self.tolerance = tolerance 
        
        #TODO temporary solution
        # self.action_space = action_space 
        # self.observation_space = observation_space 
        
        self.dimensions = dimensions
        self.control_method = control_method 
        
        super().__init__(**kwargs)
        
        if self.TASK == Task.STABILIZATION:
            self.X_GOAL = np.array(goal)
            
        elif self.TASK == Task.TRAJ_TRACKING:
            POS_REF, VEL_REF, SPEED = self._generate_trajectory(
                traj_type=self.TASK_INFO["trajectory_type"],
                traj_length=self.EPISODE_LEN_SEC,
                num_cycles=self.TASK_INFO["num_cycles"],
                traj_plane=self.TASK_INFO["trajectory_plane"],
                position_offset=self.TASK_INFO["trajectory_position_offset"],
                scaling=self.TASK_INFO["trajectory_scale"],
                sample_time=self.CTRL_TIMESTEP
            )
            if self.dimensions == 2:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 0],
                    POS_REF[:, 2],
                    np.zeros(POS_REF.shape[0]),
                    np.zeros(VEL_REF.shape[0])
                ]).transpose()
                
            elif self.dimensions == 3:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 0],
                    POS_REF[:, 1],
                    POS_REF[:, 2],
                ]).transpose()
            
        self.U_GOAL = np.array([250]) 
        self.GRAVITY_ACC = 9.8
        
        self.ACTION_LABELS = ['T']
        self.ACTION_UNITS = ['Nm']
        self.STATE_LABELS = ['theta', 'theta_dot']
        self.STATE_UNITS = ['rad', 'rad/s']
        
        self._setup_symbolic()

        # self.state = self._get_observation()
        self.JOINT_ANGULAR_POSITION_INDEX = 0
        self.JOINT_ANGULAR_VELOCITY_INDEX = 1
        self.state = self._get_symbolic_state()
        
    #TODO wanna make this as flexible? use diff action and obs space?
    def _set_action_space(self, ):
        self.action_space = spaces.Box(-3.5, 3.5, (len(self.controlled_joint_indices),), dtype=np.float32)
        
    def _set_observation_space(self):
        dim_dict = {
            "position":3 + 3,
            "orientation" : 4+4
        }
        self.observation_space = spaces.Box(-np.inf, np.inf, (dim_dict[self.observed_link_state_keys[0]], ), dtype=np.float32)
    
    def _joint_apply_action(self, joint_index, action):
        if self.controlled_variable == p.TORQUE_CONTROL:
            self._pb_client.setJointMotorControl2(self.robot, jointIndex=joint_index,
                    controlMode= self.controlled_variable, force=action
            )
        elif self.controlled_variable == p.POSITION_CONTROL:
            self._pb_client.setJointMotorControl2(self.robot, jointIndex=joint_index,
                    controlMode= self.controlled_variable, targetPosition = action
            )
        
    def _joint_get_state(self, joint_index):
        joint_state =  self._pb_client.getJointState(
            self.robot,
            jointIndex=joint_index,
        )
        return joint_state
    
    def _get_symbolic_state(self):
        # double typecasting
        joint_states = self._joint_get_state(self.controlled_joint_indices[0])
        angular_position = joint_states[self.JOINT_ANGULAR_POSITION_INDEX]
        angular_velocity = joint_states[self.JOINT_ANGULAR_VELOCITY_INDEX]
        state = np.array([angular_position, angular_velocity])
        return state 
    
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
        obs = np.array(self.link_states[link_state_key][link_index])
        
        if self.dimensions == 2:
            # take x, z 
            obs = obs[[0,2]]
        
        # TODO reuse this if want to have multiple goals and goal keys in the future. simplify for now
        # goal = [{
        #     "position": [None for i in range(13)]
        # }]
        # goal[0]["position"][observed_link_indices[0]] = position_goal
        # goal = np.array(self.goal[0][self.observed_link_state_keys[0]][link_index])
        
        # only use this for RL for now?
        
        if self.control_method == "rl":
            goal = self.goal 
            obs = list(obs) 
            
            for i in goal:
                obs.append(i)
        
            obs = np.array(obs)
        # ValueError: Error: Unexpected observation shape (4,) for Box environment, please use (1, 4) or (n_env, 1, 4) for the observation shape.
        # obs = np.expand_dims(obs, axis=0)
        #TODO observation space might vary depending on size of observed_link_indices observed_link_state_key
        
        
        return obs 
        
    def _get_reward(self):
        # TODO generalize so that can read dict goal for multiple state keys and links
        # trajectory 
        
        reward = 0 
        
        if self.COST == Cost.RL_REWARD:
            for state_key in self.observed_link_state_keys:
                for link_index in self.observed_link_indices:
                    goal = np.array(self.goal)
                    # goal = np.array(self.goal[0][state_key][link_index]) #TODO this is for point tracking only
                    if self.dimensions == 2:
                        state = np.array(self.link_states[state_key][link_index])[[0,2]]
                    elif self.dimensions == 3: 
                        state = np.array(self.link_states[state_key][link_index])
                    # rmse = np.sqrt(np.mean((goal-state)**2))
                    loss = np.sum(abs(goal-state))
                    reward -= loss
            reward = reward*1000
            
        elif self.COST == Cost.QUADRATIC:
            reward = float(-1 * self.symbolic.loss(x=self.state,
                    Xr=self.X_GOAL,
                    u=self.current_action,
                    Ur=self.U_GOAL,
                    Q=self.Q,
                    R=self.R)["l"])
        
        return reward 

    def _get_done(self):
        #TODO time limit, boundary limit
        reward = self._get_reward()
        
        if self.control_method=="rl":
            if reward > self.tolerance:
                return True 
            else:
                return False 
            
        elif self.control_method=="classical":
            if self.TASK == Task.STABILIZATION:
                # print(self.TASK_INFO)
                # for some reason overwritten by ilqr.yaml
                # if reward > -0.3 :
                #     return True 
                # else:
                #     return False 
                # print("goal")
                # print(np.linalg.norm(self.state - self.X_GOAL))
                tol = 0.8
                self.goal_reached = bool(np.linalg.norm(self.state - self.X_GOAL) < tol)
                if self.goal_reached:
                    return True
            else: 
                raise Exception("not implemented")
    def step(self, action_list:np.array):
        
        if self.target_space == "joint":
            assert len(action_list) == len(self.controlled_joint_indices), "size of action_list not equal to controlled_joint_indices"
            for action_list_index, joint_index in enumerate(self.controlled_joint_indices):
                action = action_list[action_list_index]
                if action!=None: 
                    if self.control_method == "rl":
                        applied_action = self._action_mapping_torque(action) 
                    else:
                        applied_action = action  
                self._joint_apply_action(joint_index, applied_action)
            
            self._pb_client.stepSimulation()
    
        # TODO expand this state to multiple
        # currently using this for classical control only 
        # state should correspond to x in symbolic model
        self.state = self._get_symbolic_state()
        self.current_action = action_list
    
        obs = self._get_observation()
        reward = self._get_reward()
        # print(reward)
        info = {} 
        done = self._get_done() 
        
        # print(f"state: {[round(i,3) for i in self.state]}")
        # print(f"obs: {obs}")
        # print(f"goal: {self.goal}")
        # print(f"action: {action_list}")
        # print(f"reward: {reward}")
        return obs, reward, done, info 
    
    def _action_mapping_torque(self, policy_action):
        # apply action reprojection 
        if policy_action>=0 and policy_action<=self.MIN_TORQUE_POLICY:
            applied_action = self.MIN_TORQUE_POLICY * 100
        elif policy_action<=0 and policy_action>=-self.MIN_TORQUE_POLICY:
            applied_action = -self.MIN_TORQUE_POLICY * 100
        elif policy_action > self.MIN_TORQUE_POLICY or policy_action < -self.MIN_TORQUE_POLICY:
            applied_action = policy_action  * 100 
        
        return applied_action
            
    def reset(self):
        #TODO are these enough? 
        p.resetSimulation()
        
        self.robot = p.loadURDF(self.urdf_path)
        self.state = self._get_symbolic_state()
        obs = self._get_observation()
        # obs = np.array([0.0, 0.0, 0.0, 0.0]) #TODO change to a reasonable reset point
        return obs

    def close(self):
        """Clean up the environment and PyBullet connection.

        """
        # if self.PYB_CLIENT >= 0:
        self._pb_client.__del__()
        print("close connection")
        sleep(0.5)
        # self.PYB_CLIENT = -1
        
    def _setup_symbolic(self):
        if self.controlled_variable == p.TORQUE_CONTROL:
            # single joint 
            
            # use pendulum formula for now

            g = self.GRAVITY_ACC
            dt = self.CTRL_TIMESTEP
            
            # TODO infer from URDF 
            mc = 0.5
            ma = 0.5
            
            # TODO express in terms of actual length (distance b.w 2 joints)
            # can use xyz from URDF then compute euclidean distance?

            l = 1
            I = 0 # moment of inertia from gears and motors

            # must match dimension of X_GOAL and U_GOAL
            nx = 2
            nu = 1 

            # x = cs.MX.sym('x')

            x_x = cs.MX.sym('x_x')
            # x_y = cs.MX.sym('x_y')
            x_z = cs.MX.sym('x_z')
            theta = cs.MX.sym('theta')
            theta_dot = cs.MX.sym('theta_dot')
            
            U = cs.MX.sym('U') # torque
            # theta_dot_dot = cs.MX.sym("tdotdot")
            
            # dim of X_dot have ot match X
            X_dot = cs.vertcat(theta_dot, (U-(mc/2+ma) * g * l * cs.sin(theta)) / ((mc/3 + ma)* l**2 + I))
            # Theta_dot_dot = cs.MX.sym("ttt")
            # not sure
            # X = cs.vertcat(theta)
            X = cs.vertcat(theta, theta_dot)
            Y = cs.vertcat(theta, theta_dot)

            Q = cs.MX.sym('Q', nx, nx)
            R = cs.MX.sym('R', nu, nu)

            Xr = cs.MX.sym('Xr', nx, 1)
            Ur = cs.MX.sym('Ur', nu, 1)

            # convert angular to cartesian position 
            # x_x = l * cs.cos(theta)
            # x_z = l * cs.sin(theta) 

            # X_array = cs.vertcat(x_x, x_z)

            # angular 
            cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) 
            
            # cartesian
            # cost_func = 0.5 * (X_array - Xr).T @ Q @ (X_array - Xr) 
            # + 0.5 * (cs.MX.fabs(U) - Ur).T @ R @ (cs.MX.fabs(U) - Ur)

            dynamics = {"dyn_eqn": X_dot, "obs_eqn": Y, "vars": {"X": X, "U": U}}
            cost = {"cost_func": cost_func, "vars": {"X": X, "U": U, "Xr": Xr, "Ur": Ur, "Q": Q, "R": R}}
            self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt)
            