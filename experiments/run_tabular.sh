mkdir -p logs

nohup python3 domias_main.py --seed 0 --gan_method TVAE --dataset housing --density_estimator bnaf --batch_dim 20 --training_size_list 500 --held_out_size_list 50 --gen_size_list 10000 --training_epoch_list 2000 >> logs/TVAE_final_logs1.txt &
nohup python3 domias_main.py --seed 1 --gan_method TVAE --dataset housing --density_estimator bnaf --batch_dim 20 --training_size_list 500 --held_out_size_list 50 --gen_size_list 10000 --training_epoch_list 2000 >> logs/TVAE_final_logs2.txt &
nohup python3 domias_main.py --seed 2 --gan_method TVAE --dataset housing --density_estimator bnaf --batch_dim 20 --training_size_list 500 --held_out_size_list 50 --gen_size_list 10000 --training_epoch_list 2000 >> logs/TVAE_final_logs3.txt &
nohup python3 domias_main.py --seed 3 --gan_method TVAE --dataset housing --density_estimator bnaf --batch_dim 20 --training_size_list 500 --held_out_size_list 50 --gen_size_list 10000 --training_epoch_list 2000 >> logs/TVAE_final_logs4.txt &
nohup python3 domias_main.py --seed 4 --gan_method TVAE --dataset housing --density_estimator bnaf --batch_dim 20 --training_size_list 500 --held_out_size_list 50 --gen_size_list 10000 --training_epoch_list 2000 >> logs/TVAE_final_logs5.txt &
nohup python3 domias_main.py --seed 5 --gan_method TVAE --dataset housing --density_estimator bnaf --batch_dim 20 --training_size_list 500 --held_out_size_list 50 --gen_size_list 10000 --training_epoch_list 2000 >> logs/TVAE_final_logs6.txt &
nohup python3 domias_main.py --seed 6 --gan_method TVAE --dataset housing --density_estimator bnaf --batch_dim 20 --training_size_list 500 --held_out_size_list 50 --gen_size_list 10000 --training_epoch_list 2000 >> logs/TVAE_final_logs7.txt &
nohup python3 domias_main.py --seed 7 --gan_method TVAE --dataset housing --density_estimator bnaf --batch_dim 20 --training_size_list 500 --held_out_size_list 50 --gen_size_list 10000 --training_epoch_list 2000 >> logs/TVAE_final_logs8.txt &
