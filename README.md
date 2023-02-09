# Code for DOMIAS

1. **Experiments main paper**

To reproduce results for DOMIAS, baselines, and ablated models, run
```python
python3 domias_main.py --seed 0 --gpu_idx 0 --flows 5 --gan_method TVAE --dataset housing --training_size_list 30 50 100 300 500 1000 --held_out_size_list 10000 --gen_size_list 10000 --training_epoch_list 2000
```
changing arguments training_size_list, held_out_size_list, gen_size_list, and training_epoch_list for specific experiments over ranges (Experiments 5.1 and 5.2, see Appendix A for details) and gan_method for generative model of interest.

or equivalently, run
```python
bash run_tabular.sh
```

2. **Experiments no prior knowledge (Appendix D)**

If using prior knowledge (i.e., no reference dataset setting), add
```python
--density_estimator prior
```

3. **Experiment images (Appendix B.3)**

To run experiment with the CelebA dataset, first run
```python
python3 celeba_gen.py --gpu_idx 0 --seed 0 --training_size 4000
```
and then
```python
python3 celeba_eval.py --gpu_idx 0 --seed 0 --training_size 4000
```
