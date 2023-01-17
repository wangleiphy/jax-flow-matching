import subprocess 
import numpy as np 
import time 

nickname = 'hydrogen'

###############################
model = 'ferminet'
nlayers = 4
h1size = 256
h2size = 64
nheads = 8
keysize = 16 
batchsize = 100
lr = 1e-3

###############################
prog = '../src/main.py'
resfolder = '/data/wanglei/lj/' + nickname  + '/' 

def submitJob(bin,args,jobname,logname,run=False,wait=None):

    #prepare the job file 
    job='''#!/bin/bash -l
#SBATCH --partition=a400
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --error=%s'''%(jobname,logname,logname)

    if wait is not None:
        dependency ='''
#SBATCH --dependency=afterany:%d\n'''%(wait)
        job += dependency 

    job += '''
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes:"
echo $SLURM_JOB_NODELIST
echo "Using $SLURM_NTASKS_PER_NODE tasks per node"
echo "A total of $SLURM_NTASKS tasks is used"\n
echo Job started at `date`\n'''

    job +='python '+ str(bin) + ' '
    for key, val in args.items():
        job += '--'+str(key) + ' '+ str(val) + ' '
    job += '''
echo Job finished at `date`\n'''

    #print job
    jobfile = open("jobfile", "w")
    jobfile.write("%s"%job)
    jobfile.close()

    #submit the job 
    if run:
        cmd = ['sbatch', 'jobfile']
        time.sleep(0.1)
    else:
        cmd = ['cat','jobfile']

    subprocess.check_call(cmd)
    return None

