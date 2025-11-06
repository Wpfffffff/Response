# source /fs2/home/yangtihao/packages/setenv1.sh

source ~/packages/setenv1.sh
sbatch -N 1 --ntasks-per-node=32 -t 3-20:00:00 -p xahcnormal -J Grass_RE ./run.sh   #10-00:00:00


