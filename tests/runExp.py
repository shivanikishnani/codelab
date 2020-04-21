import csv
from multiprocessing import Process
import GPUtil
import subprocess
from subprocess import Popen, PIPE
import os
from subprocess import Popen, PIPE
import glob

#find out which cpus are free and change environment variable accordingly 
deviceIDs = GPUtil.getAvailable(order = 'first', limit = 3, maxLoad = 0.5, maxMemory = 0.1, \
includeNan=False, excludeID=[], excludeUUID=[])

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

#read in dictionary file for all experiments
reader = csv.reader(open('experiments.csv', 'r'))
d = {}
for row in reader:
   k, v = row
   d[k] = v

def run_python(process):
	for exp in experiments:
		env, alg, num_timestep, save_path, log_path, network, save_video_interval = exp
		DEVICE_ID_LIST = GPUtil.getFirstAvailable()
		DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
		os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID) 
    #use subprocess here?
    #load in the commands here
    # cmds_list = [['./srun.py --env{} ..', file_name] for file_name in f_list]
    # procs_list = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmds_list]
    # for proc in procs_list:
    #   proc.wait()    
    #   f.seek(0)
    # check if subprocess is still alive 
    #     poll = p.poll()
    # if poll == None:
    # p.subprocess is alive

    os.system('python {} --env={} --alg={}, --num_timestep={} --save_path={} --log_path={} --network={} --save_video_interval={}'.format(process, env, alg))

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

if __name__ == '__main__':
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()

#read file values 

di = {'Python' : '.py', 'C++' : '.cpp', 'Java' : '.java'}
w = csv.writer(open("output.csv", "w"))
for key, val in di.items():
	w.writerow([key, val])
                 
'''
child_processes = []
for work, filename in worklist:
    with io.open(filename, mode='wb') as out:
        p = subprocess.Popen(work, stdout=out, stderr=out)
        child_processes.append(p)    # start this one, and immediately return to start another

# now you can join them together
for cp in child_processes:
    cp.wait()  
'''


