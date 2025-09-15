# MCDC
In this folder, we provide the code for MCDC and various baseline models to ensure the authenticity and reproducibility of the paper.


For various baseline models, their environment configurations follow the original paper. Specific parameters and code are provided by this folder.


For MCDC, its environment configuration and training method are consistent with Simcc. Specific parameters are provided in the 'config' folder.


Train the teacher model before training the student model. Follow the training guidelines of Simcc for the teacher model. Before training the student model, modify the path of the teacher model in the configuration file.

## Train
To train a model for COCO or MPII datasets, run:

```
python tools/train.py --cfg experiments/coco/hrnet/w32_64x64_adam_lr1e-3.yaml
```



## Test
To test a model for COCO or MPII datasets, run:

```
python tools/test.py --cfg experiments/coco/hrnet/w32_64x64_adam_lr1e-3.yaml
```




