# Deep Graphical Feature Learning for the Feature Matching Problem

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
pip install tqdm
```

