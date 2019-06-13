# SR-GNN
Implementation of ["Session-based Recommendation with Graph Neural Networks"](https://arxiv.org/pdf/1811.00855.pdf) in Tensorflow.

There are two datasets used in the paper. 
- Yoochoose: <http://2015.recsyschallenge.com/challenge.html>
- Diginetica: <http://cikm2016.cs.iupui.edu/cikm-cup>

Since the link for yoochose dataset does not work in the original repo you can download the data

```bash
curl -Lo yoochoose-data.7z https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z
7z x yoochoose-data.7z

```

After the download please put the folders under the datasets/ and rename them as only "yoochoose" and "diginetica". After moving the folders recall the preprocess script. 

##### Acknowledgments

Official implementation and the paper can be found [here](https://github.com/CRIPAC-DIG/SR-GNN). 
Borrowed the data preprocessing from original repository, including diginetica and yoochoose.

config/dev_config.yml: change the dataset name which you want to train model for.  


