# SR-GNN
A reimplementation of ["Session-based Recommendation with Graph Neural Networks"](https://arxiv.org/pdf/1811.00855.pdf) in Tensorflow.

Official implementation and the paper can be found [here](https://github.com/CRIPAC-DIG/SR-GNN). 

Borrowed the data preprocessing from original repository, including diginetica and yoochoose.

There are two datasets used in the paper. 
- Yoochoose 
- Diginetica

Since the link for yoochose dataset does not work in the original repo you can download the data

curl -Lo yoochoose-data.7z https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z
7z x yoochoose-data.7z


7-Zip [64] 9.20  Copyright (c) 1999-2010 Igor Pavlov  2010-11-18
p7zip Version 9.20 (locale=C,Utf16=off,HugeFiles=on,8 CPUs)

Processing archive: yoochoose-data.7z

Extracting  yoochoose-buys.dat
Extracting  yoochoose-clicks.dat
Extracting  yoochoose-test.dat
Extracting  dataset-README.txt

Everything is Ok

Files: 4
Size:       1914111754
Compressed: 287211932

