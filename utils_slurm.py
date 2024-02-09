import os
import warnings
import sys
sys.path.append('./')
from utils_io import make_dir

def submit_slurm(python_script, job_file,job_out_dir = '', conda_env='a100', partition='gpu',mem=32, time_hrs = -1, n_gpu = 1, n_cpu = 1, exclude_nodes = None, job_name = 'script', prioritize_cpu_nodes = True, extra_line='', nodelist = None, gpu_type = 'v100-sxm2', requeue = False):
    '''
    submit batch job to slurm

    args:
        exclude_nodes: list of specific nodes to exclude
    '''
    dname = os.path.dirname(python_script.split(' -')[0]) # cut off script name before argparse options. This is to prevent issues when providing a path as a CLI argument.
    if job_out_dir == '':
        job_out = os.path.join(dname, 'job_out')
    else:
        job_out = os.path.join(job_out_dir, 'job_out')
    make_dir(job_out)  # create job_out folder

    if partition not in ['gpu', 'short', 'ai-jumpstart']:
        raise ValueError('invalid partition specified')

    # default time limits
    time_default = {
        'gpu': 8,
        'short':24,
        'ai-jumpstart':24
    }
    # max time limits
    time_max = {
        'gpu': 8,
        'short':24,
        'ai-jumpstart':48
    }
    if time_hrs == -1:
        # set to default time limit
        time_hrs = time_default[partition]
    elif time_hrs > time_max[partition]:
        # set to maximum time limit if exceeded
        time_hrs = time_max[partition]
        warnings.warn('time limit set to maximum for %s partiton: %s hours' % (partition, str(time_hrs)))
    elif time_hrs < 0:
        raise ValueError('invalid (negative) time specified')

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("\n")
        fh.writelines("#SBATCH --job-name=%s\n" % (job_name))
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --tasks-per-node=1\n")
        fh.writelines("#SBATCH --cpus-per-task=%s\n" % str(n_cpu))
        fh.writelines("#SBATCH --mem=%sGb \n" % str(mem))
        fh.writelines("#SBATCH --output=" + job_out + "/%j.out\n")
        fh.writelines("#SBATCH --error=" + job_out + "/%j.err\n")
        fh.writelines("#SBATCH --partition=%s\n" % (partition))
        # fh.writelines("#SBATCH --nodelist=d3159")
        fh.writelines("#SBATCH --time=%s:00:00\n" % (str(time_hrs)))
        if nodelist is not None:
            fh.writelines("#SBATCH --nodelist=%s\n" % (nodelist))

        # exclude specific nodes
        if exclude_nodes is not None:
            exclude_str = ','.join(exclude_nodes)
            fh.writelines("#SBATCH --exclude=d[%s]\n" % (exclude_str))

        # specify gpu
        if partition == 'gpu':
            fh.writelines("#SBATCH --gres=gpu:%s:1\n" % gpu_type)
        elif partition == 'ai-jumpstart':
            if n_gpu>0:
                fh.writelines("#SBATCH --gres=gpu:a100:%s\n" % (str(n_gpu)))
            elif prioritize_cpu_nodes:
                # exclude gpu nodes
                fh.writelines("#SBATCH --exclude=d[3146-3150]\n")

        fh.writelines("\n")
        fh.writelines("CONDA_BASE=$(conda info --base) ; source $CONDA_BASE/etc/profile.d/conda.sh \n")
        fh.writelines("conda activate %s\n" % conda_env)
        fh.writelines("%s\n" % extra_line)
        if requeue:
            # end job early and requeue
            endtime = int(time_hrs * 60 - 10) # end job 10 min early
            fh.writelines("timeout %sm python -u %s\n" % (str(endtime), python_script))
            fh.writelines("if [[ $? == 124 ]]\n")
            fh.writelines("then\n")
            fh.writelines("    scontrol requeue $SLURM_JOB_ID\n")
            fh.writelines("fi\n")
        else:
            fh.writelines("python -u %s" % (python_script))
    os.system("sbatch %s" %job_file)


def submit_slurm_ch(python_script, job_file, conda_env='a40', partition='linux12h',mem=32, time_hrs = -1, job_name = 'script', nodelist = None, n_cpu = 1, requeue = False):
    '''
    args:
        prioritize_cpu_nodes: if using ai-jumpstart for cpu jobs, prioritize cpu-only nodes if True.
    '''
    dname = os.path.dirname(python_script.split(' -')[0]) # cut off script name before argparse options. This is to prevent issues when providing a path as a CLI argument.
    job_out = os.path.join(dname, 'job_out')
    make_dir(job_out)  # create job_out folder

    if partition not in ['linux01', 'linux12h']:
        raise ValueError('invalid partition specified')

    # default time limits
    time_default = {
        'linux12h': 12,
        'linux01':24,
    }
    # max time limits
    time_max = {
        'linux12h': 12,
        'linux01':24,
    }

    excludelist = 'beret01.bwh.harvard.edu,beret02.bwh.harvard.edu,beret03.bwh.harvard.edu,beret04.bwh.harvard.edu,beret05.bwh.harvard.edu,beret06.bwh.harvard.edu,beret07.bwh.harvard.edu,beret08.bwh.harvard.edu,beret09.bwh.harvard.edu,beret10.bwh.harvard.edu,derby01.bwh.harvard.edu,derby02.bwh.harvard.edu,derby03.bwh.harvard.edu,derby04.bwh.harvard.edu,derby05.bwh.harvard.edu,derby06.bwh.harvard.edu,derby07.bwh.harvard.edu,derby08.bwh.harvard.edu,derby09.bwh.harvard.edu,derby10.bwh.harvard.edu,derby11.bwh.harvard.edu,homburg01.bwh.harvard.edu,homburg02.bwh.harvard.edu,homburg03.bwh.harvard.edu,homburg04.bwh.harvard.edu,homburg05.bwh.harvard.edu,homburg06.bwh.harvard.edu,homburg07.bwh.harvard.edu,homburg08.bwh.harvard.edu,homburg09.bwh.harvard.edu,homburg10.bwh.harvard.edu,mint.bwh.harvard.edu,mint02.bwh.harvard.edu,mint03.bwh.harvard.edu,mint04.bwh.harvard.edu,mint05.bwh.harvard.edu,sombrero03.bwh.harvard.edu,sombrero04.bwh.harvard.edu,sombrero05.bwh.harvard.edu,sombrero06.bwh.harvard.edu,sombrero07.bwh.harvard.edu,sombrero08.bwh.harvard.edu,sombrero09.bwh.harvard.edu,sombrero10.bwh.harvard.edu,sombrero11.bwh.harvard.edu,sombrero12.bwh.harvard.edu,sombrero13.bwh.harvard.edu,sombrero14.bwh.harvard.edu,sombrero15.bwh.harvard.edu,sombrero16.bwh.harvard.edu,sombrero17.bwh.harvard.edu,sombrero18.bwh.harvard.edu,sombrero19.bwh.harvard.edu,sombrero20.bwh.harvard.edu,stetson01.bwh.harvard.edu,stetson02.bwh.harvard.edu,stetson03.bwh.harvard.edu,stetson04.bwh.harvard.edu,stetson05.bwh.harvard.edu,stetson06.bwh.harvard.edu,stetson07.bwh.harvard.edu,stetson08.bwh.harvard.edu,stetson09.bwh.harvard.edu,stetson10.bwh.harvard.edu,stetson11.bwh.harvard.edu,stetson12.bwh.harvard.edu,stetson13.bwh.harvard.edu,stetson14.bwh.harvard.edu,stetson15.bwh.harvard.edu,stetson16.bwh.harvard.edu,stetson17.bwh.harvard.edu,stetson18.bwh.harvard.edu,stetson19.bwh.harvard.edu,stetson20.bwh.harvard.edu,stetson21.bwh.harvard.edu,stetson22.bwh.harvard.edu,stetson23.bwh.harvard.edu,stetson24.bwh.harvard.edu,toque01.bwh.harvard.edu,toque02.bwh.harvard.edu,toque03.bwh.harvard.edu,toque04.bwh.harvard.edu,toque05.bwh.harvard.edu,sombrero01.bwh.harvard.edu,oolong01.bwh.harvard.edu,sombrero02.bwh.harvard.edu'

    if time_hrs == -1:
        # set to default time limit
        time_hrs = time_default[partition]
    elif time_hrs > time_max[partition]:
        # set to maximum time limit if exceeded
        time_hrs = time_max[partition]
        warnings.warn('time limit set to maximum for %s partiton: %s hours' % (partition, str(time_hrs)))
    elif time_hrs < 0:
        raise ValueError('invalid (negative) time specified')

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("\n")
        fh.writelines("#SBATCH --job-name=%s\n" % (job_name))
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --tasks-per-node=1\n")
        fh.writelines("#SBATCH --cpus-per-task=%s\n" % str(n_cpu))
        fh.writelines("#SBATCH --mem=%sGb \n" % str(mem))
        fh.writelines("#SBATCH --output=" + job_out + "/%j.out\n")
        fh.writelines("#SBATCH --error=" + job_out + "/%j.err\n")
        fh.writelines("#SBATCH --partition=%s\n" % (partition))
        if requeue: fh.writelines("#SBATCH --requeue\n")
        #fh.writelines("#SBATCH --gpus-per-node=%s\n" % str(n_gpu))
        fh.writelines("#SBATCH --time=%s:00:00\n" % (str(time_hrs)))
        if nodelist is not None: fh.writelines("#SBATCH --nodelist=%s\n" % (nodelist))
        fh.writelines("#SBATCH --exclude=%s\n" % (excludelist))

        fh.writelines("\n")
        #fh.writelines("module load anaconda3/2022.05\n")
        fh.writelines("CONDA_BASE=$(conda info --base) ; source $CONDA_BASE/etc/profile.d/conda.sh\n")
        fh.writelines("conda activate %s\n" % conda_env)
        if requeue:
            # end job early and requeue
            endtime = int(time_hrs * 60 - 30) # end job 30 min early
            fh.writelines("timeout %sm python -u %s\n" % (str(endtime), python_script))
            fh.writelines("if [[ $? == 124 ]]\n")
            fh.writelines("then\n")
            fh.writelines("    scontrol requeue $SLURM_JOB_ID\n")
            fh.writelines("    scontrol release $SLURM_JOB_ID\n")
            fh.writelines("fi\n")
        else:
            fh.writelines("python -u %s" % python_script)
    os.system("sbatch %s" %job_file)