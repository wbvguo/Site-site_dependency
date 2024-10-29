#!/bin/bash
#$ -S /bin/bash                 # RUN ON HOFFMAN2@UCLA
#$ -cwd                         # Execute the job from the current directory
#$ -j y                         # Error stream is merged with the standard output
#$ -l h_data=8G,h_rt=24:00:00   # 8G of memory and 24 hour of runtime
#$ -t 1-8:1                     # job array from 1 to n_jobs with step 1
#$ -r n                         # job is NOT rerunable  
#$ -o joblog.$JOB_ID.$TASK_ID   # Log file, remove $TASK_ID for non-array jobs


#################### LOAD ENVIRONMENT ####################
source /u/local/Modules/default/init/bash
source /u/local/Modules/default/init/modules.sh
module load anaconda3
conda activate py38


#################### FUNCTION ####################
function log_stamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $JOB_ID.$SGE_TASK_ID on node $(hostname -s): $1"
    echo " "
}


#################### CODE SECTION START  ####################
echo "[Path/Name]: $SGE_O_WORKDIR/$JOB_NAME"
log_stamp "started"


# assign parameters to each job in the array
working_path=$HOME/iproject/Site-site_dependency/
output_path=$working_path/data/heterhmm/simn/
ja=$working_path/jobsn


# read the parameters from the job file
PARMS=($(awk "NR==$SGE_TASK_ID" $ja))
line=${PARMS[0]}
IFS=',' read -r p1 p2 w0 w1 p3 p4 n prefix <<< "$line"

echo "[TASK_LABEL]: $line"
python3 $working_path/code/sim_heterhmm.py -p1 $p1 -p2 $p2 -w0 $w0 -w1 $w1 -p3 $p3 -p4 $p4 -n $n -o $output_path -p $prefix


#################### CODE SECTION END  ####################
log_stamp "finished"


#################### EXAMPLE ####################
# qsub qjob.sh

