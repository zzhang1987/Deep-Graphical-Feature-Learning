# Deep Graphical Feature Learning for the Feature Matching Problem, ICCV2019

Created by [Zhen Zhang](https://zzhang.org) and [Wee Sun Lee](https://www.comp.nus.edu.sg/~leews/). 


## Citation

If you find the code useful, please consider citing 

```
@inproceedings{Zhang_2019_ICCV,
    Author = {Zhen Zhang and Wee Sun Lee},
    Title = {Deep Graphical Feature Learning for the Feature Matching Problem},
    Year = {2019},
    booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
}
```


## Dependencies

Please install the following dependencies for training and testing

``` shell
conda create -n python3.6 python=3.6
conda activate python3.6
conda install tensorflow
conda install conda install pytorch torchvision cudatoolkit=10.0 -c pytorch # adjust the cuda version according to your platform
conda install scikit-image
pip install tqdm
```

## Train the model 

The following code can be used to train the model:

``` shell
python train.py --batch_size 64 # on RTX2080Ti, uses about 9GB GPU memory
```


## Test the model 
After training over random generated 9M samples, the training code will finally generate a parameter file ``matching_res_True_gp_True_epoch_8.pt''. As in our code the random matching pairs are generated on the fly, it is equivalent to training over 9M samples for one epoch. 


### Synthetic data 
To reproduce the experimental results on synthetic data, please run the following script:

``` shell
python test_syn.py --param_path ./matching_res_True_gp_True_epoch_8.pt
```

### CMU House 
The original link to download the CMU house dataset is not valid anymore, thus the data is included in the repo.
To reproduce the experimental results on the dataset, please run the script as follows

``` shell
python test_cmu_house.py --param_path ./matching_res_True_gp_True_epoch_8.pt --data_path ./datasets/cmum/house
```


### PF-Pascal
To reproduce the experimental results on PF-Pascal dataset, please first download the PF-Pascal dataset by running the script as follows,

``` shell
pushd datasets
./downloads.sh
popd
```

After that, the results can be reproduces by running the following script

``` shell
python test_pascal_pf.py --param_path ./matching_res_True_gp_True_epoch_7.pt --data_path ./datasets/PF-dataset-PASCAL/ --random_rotate False 
python test_pascal_pf.py --param_path ./matching_res_True_gp_True_epoch_7.pt --data_path ./datasets/PF-dataset-PASCAL/ --random_rotate True
```

