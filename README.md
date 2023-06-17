# install
conda create -n fpm python=3.7.11

conda activate fpm

conda install pytorch=1.7 torchvision cudatoolkit=11.0 -c pytorch

pip install -r requirements.txt

# Project explanation
https://zhuanlan.zhihu.com/p/637685577

# Benchmark and model zoo
All results are reported on coco2017 val set. 

|Backbone/Neck          |mask point          |    epoch     |    size  |  AP(mask) | AP(box) | Download|
|:-------------:|:-------------:| :-------------: | :-----:| :-----: | :----: | :---------------------------------------------------------------------------------------: |
|  PPYOLOE-s-FastPolarMask    |36      |    300     |    640    |   33.1    |  32.6 |        config/model (coming soon)|
|

