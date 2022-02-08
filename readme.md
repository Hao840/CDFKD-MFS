# CDFKD-MFS

PyTorch implementation of the paper: CDFKD-MFS: Collaborative Data-free Knowledge Distillation via Multi-level Feature Sharing. It is an extended version of the ICME21 paper 'Model Compression via Collaborative Data-Free Knowledge Distillation for Edge Intelligence' and is currently under review.



## Requirements

```
pip install -r requirements.txt
```



## Quick Start

An example of distilling knowledges from 3 pre-trained ResNet-34 models into a ResNet-18 model on CIFAR-100.

1. Preparation: modify the `utils\classification_dataset\config.py` file to type in the path for saving datasets in your computer.

2. Train teacher models by running the following command for 3 times

   ```
   bash scripts\DataTrain\DatTrain-cifar100-resnet8x34.sh
   ```

3. Make a folder named `ckp` in the root path, and copy the trained models into it. The trained models are distinguish by adding a suffix in the file name, e.g., `cifar100-resnet8x34-1.pt`. Counting in the suffix starts from 1.

4. Distill the knowledge of trained teachers into a multi-header feature-sharing student

   ```
   bash scripts\CDFKD_MFS\CDFKD_MFS-cifar100-resnet8x18-mhwrn4x402.sh
   ```



## More Implementations

Please yourself to run more implementations with configurations in the `scripts` folder.



## Citation

coming soon



## Reference

[CDFKD](https://github.com/Hao840/pytorch-CDFKD)

[CMI](https://github.com/zju-vipa/CMI?utm_source=catalyzex.com)

[DFAD](https://github.com/VainF/Data-Free-Adversarial-Distillation)

[DeepInversion](https://github.com/NVlabs/DeepInversion)