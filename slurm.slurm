#!/bin/bash -l
# BEGIN SLRUM OPTIONS DECLARATIONS
# Export all environment variables

#SBATCH --export=ALL
#SBATCH --export=TMPDIR=/tmp
#SBATCH --propagate=NONE
#SBATCH --time=2-00:00:00
# Your job name
#SBATCH --job-name=testrun
#
# Use current working directory
#SBATCH --chdir=/fs1/home/ceoas/leendere/V1
#
# Set queue to use
#SBATCH --partition=ceoas-arm
#
#SBATCH --nodelist=youmu
#
#Output files for stdout and stderr
#SBATCH --output=testrun/testrun.o
#SBATCH --error=testrun/testrun.e
#
# Send an email at the beginning or end of the job (b and e), or if it's aborted or rescheduled (a)
#SBATCH --mail-user=evanleendertse@gmail.com
#SBATCH --mail-type=ALL
#
# END SLURM OPTIONS DECLARATIONS
#
PATH=/home/ceoas/leendere/miniconda3_x86/envs/terramind/bin:/home/ceoas/leendere/miniconda3_x86/condabin:/local/cluster/sge/bin:/local/cluster/sge/bin/lx-amd64:/usr/local/cuda/bin:/home/ceoas/leendere/scripts:/home/ceoas/leendere/bin:/local/ceoas/bin:/local/ceoas/x86_64/bin:/local/cluster/slurm/bin:/local/cluster/MATLAB/R2023b/bin:/bin:/usr/bin:/usr/local/bin:/usr/X11R6/bin:/usr/X/bin:.
LD_LIBRARY_PATH=/local/ceoas/x86_64/lib:/local/ceoas/x86_64/lib64
export PATH
export R_LIBS=/home/ceoas/leendere/R
#
#
#The following auto-generated commands will be run by the execution node.
#We execute your command via /usr/bin/time with a custom format
#so that the memory usage and other stats can be tracked; note that
#GNU time v1.7 has a bug in that it reports 4X too much memory usage
source /home/ceoas/leendere/miniconda3_x86/etc/profile.d/conda.sh
conda activate terramind
echo "  Started on:           " `/bin/hostname -s` 
echo "  Started at:           " `/bin/date` 
/usr/bin/time -f " \\tFull Command:                      %C \\n\\tMemory (kb):                       %M \\n\\t# SWAP  (freq):                    %W \\n\\t# Waits (freq):                    %w \\n\\tCPU (percent):                     %P \\n\\tTime (seconds):                    %e \\n\\tTime (hh:mm:ss.ms):                %E \\n\\tSystem CPU Time (seconds):         %S \\n\\tUser   CPU Time (seconds):         %U " \
python3 train_experiment.py
echo "  Finished at:           " `date` 
