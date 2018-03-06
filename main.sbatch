#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=59:59:00
#SBATCH --output=main_out.out

python3 main.py
