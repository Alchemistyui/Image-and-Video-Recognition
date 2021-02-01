#!/bin/sh
#$ -cwd

cd /gs/hs0/tga-shinoda/20M38216/final_project/
source /home/0/20M38216/.bashrc
source activate d2l


python train_resnet.py --log_dir "draw_fig/" --task_list 'Shen' 'Peng' 'Cheng' 'Wang' >& /gs/hs0/tga-shinoda/20M38216/final_project/draw_fig/ewc_out.txt
python train_no_ewc.py --log_dir "draw_fig/no_ewc/" --task_list 'Shen' 'Peng' 'Cheng' 'Wang' >& /gs/hs0/tga-shinoda/20M38216/final_project/draw_fig/no_ewc_out.txt
