### 1. Prepare the datasets.

- (1) RegDB Dataset [3]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

    - (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 

    - A private download link can be requested via sending me an email (mangye16@gmail.com). 
  
- (2) SYSU-MM01 Dataset [4]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Training.
  Train a model by
  ```bash
python train.py --dataset sysu --lr 0.1 --method CmCEN --gpu 1
```

  - `--dataset`: which dataset "sysu" or "regdb".

  - `--lr`: initial learning rate.
  
  -  `--method`: method to run or baseline.
  
  - `--gpu`:  which gpu to run.

You may need mannully define the data path first.

**Parameters**: More parameters can be found in the script.

**Sampling Strategy**: N (= bacth size) person identities are randomly sampled at each step, then randomly select four visible and four thermal image. Details can be found in Line 302-307 in `train.py`.

**Training Log**: The training log will be saved in `log/" dataset_name"+ log`. Model will be saved in `save_model/`.

### 3. Testing.

Test a model on SYSU-MM01 or RegDB dataset by 
  ```bash
python test.py --mode all --resume 'model_path' --gpu 1 --dataset sysu
```
  - `--dataset`: which dataset "sysu" or "regdb".
  
  - `--mode`: "all" or "indoor" all search or indoor search (only for sysu dataset).
  
  - `--trial`: testing trial (only for RegDB dataset).
  
  - `--resume`: the saved model path.
  
  - `--gpu`:  which gpu to run.

### 4. Citation

Please kindly cite this paper in your publications if it helps your research:
```
@inproceedings{DBLP:conf/iconip/XuWLX21,
  author    = {Xiaohui Xu and
               Song Wu and
               Shan Liu and
               Guoqiang Xiao},
  editor    = {Teddy Mantoro and
               Minho Lee and
               Media Anugerah Ayu and
               Kok Wai Wong and
               Achmad Nizar Hidayanto},
  title     = {Cross-Modal Based Person Re-identification via Channel Exchange and
               Adversarial Learning},
  booktitle = {Neural Information Processing - 28th International Conference, {ICONIP}
               2021, Sanur, Bali, Indonesia, December 8-12, 2021, Proceedings, Part
               {I}},
  series    = {Lecture Notes in Computer Science},
  volume    = {13108},
  pages     = {500--511},
  publisher = {Springer},
  year      = {2021},
  url       = {https://doi.org/10.1007/978-3-030-92185-9\_41},
  doi       = {10.1007/978-3-030-92185-9\_41},
  timestamp = {Tue, 14 Dec 2021 17:56:34 +0100},
  biburl    = {https://dblp.org/rec/conf/iconip/XuWLX21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


