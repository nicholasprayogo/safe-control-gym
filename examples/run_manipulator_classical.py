import os 
from functools import partial
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
import yaml

fac = ConfigFactory()
config = fac.merge()

config.algo = "mpc"
config.task = "manipulator"
controller_yaml_path = "safe_control_gym/controllers/mpc/mpc.yaml"
env_yaml_path = "safe_control_gym/envs/manipulators/manipulator.yaml"

# cwd = os.path.join(home, "src/interpretable_ts_clustering/")

with open(controller_yaml_path, "r") as yamlfile:
    config.algo_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
with open(env_yaml_path, "r") as yamlfile:
    config.task_config = yaml.load(yamlfile, Loader=yaml.FullLoader)


env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
control_agent = make(config.algo,
            env_func,
            training=True,
            checkpoint_path=os.path.join(config.output_dir, "model_latest_manippulator.pt"),
            output_dir=config.output_dir,
            device=config.device,
            seed=config.seed,
            **config.algo_config)

# reset env 
control_agent.env.reset() 
print(control_agent.env.constraints.constraints[0].constraint_filter)

# reset controller
control_agent.reset()
# run controller
control_agent.run()