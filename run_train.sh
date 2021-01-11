#!/bin/sh
#$ -cwd

cd /gs/hs0/tga-shinoda/20M38216/final_project/
source /home/0/20M38216/.bashrc
source activate d2l


for i in $(seq 1 5)
do
    python train_resnet.py --log_dir "new_logs/Wang_Shen_Peng_Cheng/exp_"${i} --task_list 'Wang' 'Shen' 'Peng' 'Cheng' >& /gs/hs0/tga-shinoda/20M38216/final_project/cmd_out/Wang_Shen_Peng_Cheng/exp${i}_out.txt
    python train_resnet.py --log_dir "new_logs/Shen_Wang_Peng_Cheng/exp_"${i} --task_list 'Shen' 'Wang' 'Peng' 'Cheng' >& /gs/hs0/tga-shinoda/20M38216/final_project/cmd_out/Shen_Wang_Peng_Cheng/exp${i}_out.txt
    python train_resnet.py --log_dir "new_logs/Shen_Peng_Wang_Cheng/exp_"${i} --task_list 'Shen' 'Peng' 'Wang' 'Cheng' >& /gs/hs0/tga-shinoda/20M38216/final_project/cmd_out/Shen_Peng_Wang_Cheng/exp${i}_out.txt
    python train_resnet.py --log_dir "new_logs/Shen_Peng_Cheng_Wang/exp_"${i} --task_list 'Shen' 'Peng' 'Cheng' 'Wang' >& /gs/hs0/tga-shinoda/20M38216/final_project/cmd_out/Shen_Peng_Cheng_Wang/exp${i}_out.txt
done


