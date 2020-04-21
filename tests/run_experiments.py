import shutil
import os
from datetime import datetime
import pickle
from IPython import embed
import pdb
import argparse
import multiprocessing
#from tensorflow.python.client import device_lib
from datetime import date
import time
import GPUtil
import re

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def make_dir(folder, typeE):
    if not os.path.exists(folder):
        print('Making directory: {}'.format(folder))
        os.makedirs(folder)
        return folder
    else:
        print('Existing directory: {}'.format(folder))
        folder_name = folder.split('/')[-1]
        count = sum([folder_name in name for name in os.listdir('/home/shivanik/lab/' + typeE)])
        folder = folder + "_" + str(count + 1)
        return make_dir(folder, typeE)

 
# # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def set_gpu():
    gpu = GPUtil.getAvailable('last', limit=5, excludeID=[0, 1])
    vis_gpu = ""
    for g in gpu:
        vis_gpu += ", " + str(g)
    vis_gpu = vis_gpu[1:]
    os.environ["CUDA_VISIBLE_DEVICES"] = vis_gpu

def run_experiment(alg, env, save_path, log_path, num_iter, save_video_interval, save_video_length, network, gpus, load_path, reward_scale):
    if load_path is "" or load_path == "":
        str_prog = 'python ~/srun.py --alg={} --env={} --save_path={} --log_path={} --num_timesteps={}\
          --save_video_interval={} --save_video_length={} --network={}'.format(
            alg, 
            env,
            save_path, 
            log_path, 
            num_iter, 
            save_video_interval, 
            save_video_length,
            network
            )
    else:
        str_prog = 'python ~/srun.py --alg={} --env={} --load_path={} --log_path={} --num_timesteps={}\
          --save_video_interval={} --save_video_length={} --network={} --reward_scale={}'.format(
            alg, 
            env,
            load_path, 
            log_path, 
            num_iter, 
            save_video_interval, 
            save_video_length,
            network,
            reward_scale
            )
    os.system(str_prog)
 

def get_exp_name(env, alg, typeE, num_iter, network): 
    func = lambda x: num_iter // x 
    round_func = lambda x: round(num_iter / x, 2)

    if func(1e9) > 0:
        num_iter = str(round_func(1e9)) + "G"
    elif func(1e6) > 0:
        num_iter = str(round_func(1e6)) + "M"
    elif func(1e3) > 0:
        num_iter = str(round_func(1e3)) + "K"
    else:
        num_iter = str(num_iter)

    return env[:-3] + '_' + typeE + '_' + alg +  '_' + num_iter + '_' + network 

def get_env_name(env, num, network):
    env = re.sub("[0-9]-", str(num)+"-", env)
    if 'cnn' in network.lower():
        env = re.sub("[0-9]-", str(num)+"-Latent-", env)
    return env

def get_env_type(env):
    env_type = re.match('[A-Z].*?\w(?=[A-Z])', env) #eg fetch or point
    env_type = env_type.group()
    return env_type.lower()

EXP_DATA_PATH = '/home/shivanik/lab'
LOG_DIR = 'logs'
VIDEO_DIR = 'video'
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-env", type=str, default='PointMass-1-v1',
        help='Name of environment')
    parser.add_argument("-low", type=int, default=1,
        help='Lower bound on number of blocks to test for')
    parser.add_argument("-high", type=int, default=2,
        help='Upper bound on number of blocks to test for (inclusive)')
    parser.add_argument("-algs", nargs='+', type=str, 
        help='List of algorithms to run', 
        default=['ppo2']) #, 'td3', 'sac', 'ddpg'])
    parser.add_argument('-iter', nargs='+', type=int, 
        help='List of iterations to run each algorithm for', default=[5000000])
    parser.add_argument("-np", "--n_proc", type=int, default=4,
        help='Number of prcoesses')
    parser.add_argument("-typeE", type=str, default="state",
        help='Type of experiments to run, eg: state, latent')
    parser.add_argument("-vid_interval", "--save_video_interval", type=int, default=500000,
        help='Type of experiments to run, eg: state, latent')
    parser.add_argument("-vid_len", "--save_video_length", type=int, default=100,
        help='Type of experiments to run, eg: state, latent')
    parser.add_argument("-network", "--network", type=str, default='MlpPolicy',
        help='network type (MlpPolicy, CnnPolicy, LnCnnPolicy, LnMlpPolicy)')
    parser.add_argument("-gpu", nargs='+', type=int, default=[4, 5, 6, 7],
        help='List of GPUs to use for the experiments')
    parser.add_argument("-reward_scale", nargs='+', type=int, default=[0.1, 0.2, 0.5],
        help='List of GPUs to use for the experiments')
    parser.add_argument("-load_path", type=str, default="",
        help='Whether to load path or not')
    parser.add_argument("-algs", nargs='+', type=str, 
        help='List of algorithms to run', 
        default=['ppo2'])


    args, unknown = parser.parse_known_args()

    pool = multiprocessing.Pool(processes=args.n_proc)

    print(args.algs)
    print(args.iter)
    if not (len(args.iter) == len(args.algs) or len(args.iter) == 1):
        print("Enter same size iteration list and algorithm list")
        exit()

    low, high = args.low, args.high

    if high == -1 or high == low:
        high = low + 1

    set_gpu()
    network = args.network
    print(network)
    print("Low: %d, High: %d" %(low, high))
    typeE = args.typeE

    if typeE is "":
        typeE = "latent" if "cnn" in network.lower() else "state" 

    path = os.path.join(EXP_DATA_PATH, typeE)
    #path = make_dir(path)
    if len(args.iter) == 1:
        args.iter = args.iter * (high-low)      

    for i in range(high - low):
        for rw_scl in args.reward_scale:
            env_type = get_env_type(args.env) + str(low)
            args_type = 'all' if len(args.algs) == 4 else str(args.algs[0])
            new_path = os.path.join(path, env_type + '_' +  args_type) #+ '_' + str(rw_scl))
            new_path = make_dir(new_path, typeE) 
            l_path = os.path.join(new_path, LOG_DIR)


            for j, alg in enumerate(args.algs):
                num_iter = args.iter[0]
                env_name = get_env_name(args.env, low, network)
                exp_name = get_exp_name(env_name, typeE, alg, num_iter, network)
                save_path = os.path.join(new_path, exp_name)
                log_path = os.path.join(l_path, exp_name)       
                pool.apply_async(run_experiment, args=(alg, env_name, save_path, log_path, num_iter, 
                    args.save_video_interval, args.save_video_length, network, args.gpu, args.load_path, rw_scl))
                env_type = get_env_type(args.env)
                args_type = 'all' if len(args.algs) == 4 else str(args.algs[0])
                additional = 'dense' if 'dense' in args.env.lower() else 'sparse'
                if 'dense' in additional:
                    additional += str(reward_scale)
                
            new_path = os.path.join(path, env_type + str(low + i) + '_' +  args_type)
            l_path = os.path.join(new_path, LOG_DIR)
            make_dir(new_path, typeE) 
            for j, alg in enumerate(args.algs):
                num_obj = low + i
                num_iter = args.iter[i]
                env_name = get_env_name(args.env, num_obj, network)
                exp_name = get_exp_name(env_name, typeE, alg, num_iter, network)
                save_path = os.path.join(new_path, exp_name)
                log_path = os.path.join(l_path, exp_name)       
                pool.apply_async(run_experiment, args=(alg, env_name, save_path, log_path, num_iter, 
                    args.save_video_interval, args.save_video_length, network, args.gpu, args.load_path))

    pool.close()
    pool.join()
 
if __name__ == '__main__':
    main()


 











