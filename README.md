# Baseline codes for replication: An Open-Set Recognition and Few-Shot Learning Dataset for Audio Event Classification in Domestic Environments

New audio dataset with Few-Shot Learning and Open-Set Recognition issues. The dataset consists of 34 classes. 24 of them are considered known classes 
(classes to be classified) and the remaining 10 unknown classes (classes to be rejected). 
More information on the classes and their distribution can be found in the paper. 

Thanks to the configuration of the dataset, two different experimentation scenarios can be created. 

- The 24 known classes are to be classified. Depending on the number of unknown classes present in training, the values of openness can be 0, 0.04 or 0.09.
- The 24 known classes are divided into 8 sets of 3. Depending on the number of unknown classes present in training, openness values can be 0, 0.13 or 0.39.

The dataset is presented with two possible baselines based on transfer learning techniques. The networks used as feature extractor are L3Net and YAMNet.

## Setting environment
```
conda create -y -n fsl_osr python=3.6    
conda activate fsl_osr
pip install -r requirements.txt
```

## Replicating experimentation

Launch code as it is `python main.py`. `configuration.yaml` is set will all possible options

## Demo

For demonstration purposes, two pre-trained models have been released with 4 shot and openness 0 and 24 known classes.
In order to check its behaviour please launch `demo.py` as:

```bash
python demo.py <network> <path to wav> <path to model in h5 format>
```

*Examples:*

```bash
python demo.py yamnet demo\wavs\pattern_01_01.wav demo\pre-trained-models\yamnet.h5
```
Output: [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

```bash
python demo.py yamnet data\unwanted_sounds\car_horn\car_horn_01.wav demo\pre-trained-models\yamnet.h5
```
Output: [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

As it can be seen, a pattern sound is correctly classified with a 1 in the specific position and the unwanted/unknown
is rejected with all 0.

## Citation

Submitted to IEEEAccess
