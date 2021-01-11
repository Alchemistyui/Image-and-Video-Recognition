#!/bin/sh
#$ -cwd

cd /gs/hs0/tga-shinoda/20M38216/final_project/
source /home/0/20M38216/.bashrc
source activate d2l


for i in $(seq 1 5)
do
python train_resnet.py --log_dir "new_logs/exp_"${i} --task_list 'Shen' 'Wang' 'Peng' 'Cheng' >& /gs/hs0/tga-shinoda/20M38216/final_project/cmd_out/exp${i}_out.txt
done

