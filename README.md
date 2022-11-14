# Potent Antibiotic Design via Guided Search from Antibacterial Activity Evaluations
In this work, we report on a method for generating potent antibiotics, MDAGS. The proposed method combines two new ideas: the encoder and the predictor are trained jointly to learn the potential property space, and the attribute-guided optimization strategy is used in the potential space to make the model explore along the direction of the expected molecular properties. We compared our model with the methods of previous Moses and Guacamol datasets. In addition, the generated molecules were analyzed visually to demonstrate the effectiveness of the method.
## Install
* python=3.6
* fcd-torch=1.0.7
* matplotlib=3.3.4
* numpy=1.19.2
* pandas=1.5.1
* pip=0.40.0
* pytorch=1.10.1
* rdkit=2021.03.4
* scikit-learn=0.24.2
* scipy=1.5.4
* seaborn=0.11.2
* torchaudio=0.10.1
* torchvision=0.11.1
* tqdm=4.63.0
## Data
In order to train a model, you must provide training data containing molecules with target values.
### the main dataset
The main dataset is under the directory data
### MOSES Dataset and GuacaMol Dataset
The data for these two datasets can be downloaded from the following links
* The processed Guacamol and MOSES datasets in csv format can be downloaded from this link
```
https://drive.google.com/drive/folders/1LrtGru7Srj_62WMR4Zcfs7xJ3GZr9N4E?usp=sharing
```
* Original Moses dataset can be found here
```
https://github.com/molecularsets/moses
```
* Original Guacamol dataset can be found here
```
https://github.com/BenevolentAI/guacamol
```
## Training and generation
### Encoder-predictor
* Perform data enhancement on the training set
```
python chemprop/scripts/save_features.py --data_path <data_path> --save_path <save_path> --features_generator <features_generator>
```
* Optimize the hyperparameter
```
python chemprop/hyperparameter_optimization.py --data_path <data_path> --dataset_type regression --num_iters 20 --config_save_path <save_path>
```
* Train
```
python chemprop/train.py --data_path <data_path> --dataset_type regression --save_dir <save_dir> --no_features_scaling --num_folds 4 --quiet --config_path <config_path>
```
### Potential space optimization
```
python optimization_ffn.py
python optimization_GP.py
```
### Generator
* Pre-training
```
python GPT/train/train.py --run_name guacamol_fingerprint_pretrain --data_name guacamol2 --batch_size 128 --num_props 1 --max_epochs 10
python GPT/train/train.py --run_name moses_fingerprint_pretrain --data_name moses2 --batch_size 128 --num_props 1 --max_epochs 10
```
* Fine-tuning
```
python GPT/train/fine_tuning.py --run_name guacamol_fingerprint_fine_tuning --model_weight guacamol_fingerprint_pretrain.pt --batch_size 128 --max_epochs 20 --num_props 1
```
* Generation
```
python GPT/generate/generate.py --model_weight guacamol_fingerprint_fine_tuning.pt --props fingerprint --csv_name guacamol_generate_ffn_fine_tuning_after --gen_size 234 --batch_size 1024
```
