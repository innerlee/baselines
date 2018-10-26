"""
Helpers for scripts like run_atari.py.
"""

import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import retro_wrappers

import numpy as np
from curriculum.envs.arm3d.arm3d_move_peg_env import Arm3dMovePegEnv
from curriculum.envs.arm3d.arm3d_disc_env import Arm3dDiscEnv, Arm3dDiscEnvllx

def make_vec_env(env_id, env_type, num_env, seed, wrapper_kwargs=None, start_index=0, reward_scale=1.0, gamestate=None,render=False,play=False, \
                stepNumMax = 300,sparse1_dis=0.1, rewardModeForArm3d=None, initStateForArm3dTask2=None,task2InitNoise=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    def make_thunk(rank, indexTasks=None, envsIndex=0):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            subrank = rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            envsIndex=envsIndex,
            indexTasks=indexTasks,
            render=render,
            play=play, 
            stepNumMax = stepNumMax,
            sparse1_dis=sparse1_dis, 
            rewardModeForArm3d=rewardModeForArm3d, 
            initStateForArm3dTask2=initStateForArm3dTask2,
            task2InitNoise=task2InitNoise
        )

    set_global_seeds(seed)
    if num_env > 1: return SubprocVecEnv([make_thunk(i + start_index, indexTasks=i, envsIndex=i) for i in range(num_env)])
    else: return DummyVecEnv([make_thunk(start_index)])


def make_env(env_id,  env_type, envsIndex=0,subrank=0, seed=None, reward_scale=1.0, gamestate=None, wrapper_kwargs=None,render=False,play=False, \
                 indexTasks=None,stepNumMax = 300,sparse1_dis=0.1, rewardModeForArm3d=None, initStateForArm3dTask2=None, task2InitNoise=False):
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    if env_id == 'arm3d_task12_without':
        if indexTasks % 2 == 0:
            env = Arm3dDiscEnvllx(envIdLLX=1, stepNumMax=stepNumMax, sparse1_dis=sparse1_dis)
            env.envId = envsIndex

            env.rewardMode = rewardModeForArm3d
            env.reward_range = [0, 1]
            env.metadata = {'render.modes': []}
            env.unwrapped = None
            env._configured = None
            env.spec.id = 99
            # use the initial disc position of task2 as the goal to task1
            tmp_env = Arm3dDiscEnv()
            tmp_env.reset(init_state=initStateForArm3dTask2)
            env.goalPostitionTask1 = np.asarray(tmp_env.get_disc_position())
            env.seed(seed + subrank if seed is not None else None)

            env = Monitor(
                env,
                logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                allow_early_resets=True,
                render=render,
                play=play)
        elif indexTasks % 2 == 1:
            env = Arm3dDiscEnvllx(stepNumMax=stepNumMax, sparse1_dis=sparse1_dis, task2InitNoise=task2InitNoise)
            env.envId = envsIndex

            env.rewardMode = rewardModeForArm3d
            env.reward_range = [0, 1]
            env.metadata = {'render.modes': []}
            env.unwrapped = None
            env._configured = None
            env.spec.id = 99
            env.init_state = initStateForArm3dTask2
            env.seed(seed + subrank if seed is not None else None)

            env = Monitor(
                env,
                logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                allow_early_resets=True,
                render=render,
                play=play)
            env.arm3dTask2 = True

    elif env_id == 'arm3d_task12_with':
        if indexTasks % 3 == 0:
            env = Arm3dDiscEnvllx(envIdLLX=1, stepNumMax=stepNumMax, sparse1_dis=sparse1_dis)
            env.envId = envsIndex

            env.rewardMode = rewardModeForArm3d
            env.reward_range = [0, 1]
            env.metadata = {'render.modes': []}
            env.unwrapped = None
            env._configured = None
            env.spec.id = 99
            # use the initial disc position of task2 as the goal to task1
            tmp_env = Arm3dDiscEnv()
            tmp_env.reset(init_state=initStateForArm3dTask2)
            env.goalPostitionTask1 = np.asarray(tmp_env.get_disc_position())
            env.seed(seed + subrank if seed is not None else None)

            env = Monitor(
                env,
                logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                allow_early_resets=True,
                render=render,
                play=play)
        elif indexTasks % 3 == 1:
            env = Arm3dDiscEnvllx(stepNumMax=stepNumMax, sparse1_dis=sparse1_dis, task2InitNoise=task2InitNoise)
            env.envId = envsIndex

            env.rewardMode = rewardModeForArm3d
            env.reward_range = [0, 1]
            env.metadata = {'render.modes': []}
            env.unwrapped = None
            env._configured = None
            env.spec.id = 99
            env.init_state = initStateForArm3dTask2
            env.seed(seed + subrank if seed is not None else None)

            env = Monitor(
                env,
                logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                allow_early_resets=True,
                render=render,
                play=play)
            env.arm3dTask2 = True

        else:
            env = Arm3dDiscEnvllx(stepNumMax=stepNumMax,sparse1_dis=sparse1_dis)
            env.envId = envsIndex

            env.rewardMode = rewardModeForArm3d
            env.reward_range=[0.1, 0.9]
            env.metadata = {'render.modes': []}
            env.unwrapped = None
            env._configured = None
            env.spec.id = 99
            env.seed(seed + subrank if seed is not None else None)
            env = Monitor(
                env,
                logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                allow_early_resets=True,
                render=render,
                play=play)

    elif env_id == 'arm3d_task1':
        env = Arm3dDiscEnvllx(envIdLLX=1, stepNumMax=stepNumMax, sparse1_dis=sparse1_dis, task2InitNoise=task2InitNoise)
        env.envId = envsIndex

        env.rewardMode = rewardModeForArm3d
        env.reward_range = [0.1, 0.9]
        env.metadata = {'render.modes': []}
        env.unwrapped = None
        env._configured = None
        env.spec.id = 99
        env.seed(seed + subrank if seed is not None else None)
        # use the initial disc position of task2 as the goal to task1
        tmp_env = Arm3dDiscEnv()
        tmp_env.reset(init_state=initStateForArm3dTask2)
        env.goalPostitionTask1 = np.asarray(tmp_env.get_disc_position())
        env = Monitor(
            env,
            logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
            allow_early_resets=True,
            render=render,
            play=play)

    elif env_id == 'arm3d_task2':
        env = Arm3dDiscEnvllx(stepNumMax=stepNumMax * 2, sparse1_dis=sparse1_dis, task2InitNoise=task2InitNoise)
        env.envId = envsIndex

        env.rewardMode = rewardModeForArm3d
        env.reward_range = [0.1, 0.9]
        env.metadata = {'render.modes': []}
        env.unwrapped = None
        env._configured = None
        env.spec.id = 99
        env.init_state = initStateForArm3dTask2
        env.seed(seed + subrank if seed is not None else None)

        env = Monitor(
            env,
            logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
            allow_early_resets=True,
            render=render,
            play=play)

        env.arm3dTask2 = True

    elif env_id == 'arm3d':
        env = Arm3dDiscEnvllx(stepNumMax=stepNumMax, sparse1_dis=sparse1_dis)
        env.envId = envsIndex

        env.rewardMode = rewardModeForArm3d
        env.reward_range = [0.1, 0.9]
        env.metadata = {'render.modes': []}
        env.unwrapped = None
        env._configured = None
        env.spec.id = 99
        env.seed(seed + subrank if seed is not None else None)
        env = Monitor(
            env,
            logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
            allow_early_resets=True,
            render=render,
            play=play)
    else:
        env = make_atari(env_id) if env_type == 'atari' else gym.make(env_id)
        env.seed(seed + subrank if seed is not None else None)
        env = Monitor(
            env,
            logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
            allow_early_resets=True,
            render=render,
            play=play)

    if env_type == 'atari': return wrap_deepmind(env, **wrapper_kwargs)
    elif reward_scale != 1: return retro_wrappers.RewardScaler(env, reward_scale)
    else: return env



def make_mujoco_env(env_id, seed, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed  + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = gym.make(env_id)
    logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env

def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def mujoco_arg_parser():
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6)
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env',help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco',default=None,type=int)
    parser.add_argument('--num_env_play', type=int, default=1)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--load_path', help='Path to load trained model to', default=None, type=str)
    parser.add_argument('--load_num', type=str, default='00001')
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--rewardModeForArm3d', help='reward Mode for Arm3d', type=str, default='')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--renderMode', type=str, default='single')
    parser.add_argument('--record', default=False, action='store_true')
    parser.add_argument('--task2InitNoise', type=float, default=0.0)
    parser.add_argument('--ps', help='Some key word for save special log', type=str, default='')
    parser.add_argument('--stepNumMax', help='stepNumMax', type=int, default=300)
    parser.add_argument('--sparse1_dis', type=float, default=0.1)
    parser.add_argument('--ent_coef', type=float, default=0.00)
    parser.add_argument('--load_num_env',help='Number of environment copies being run in parallel with a loaded model',type=int,default=None)
    parser.add_argument('--interval_RecordSteps', type=int, default=100)
    parser.add_argument('--interval_VedioTimer', type=int, default=10)
    parser.add_argument('--interval_RecordUpdate', type=int, default=1)
    return parser

def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dicitonary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval
