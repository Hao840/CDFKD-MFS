# CDFKD-MFS

PyTorch implementation of the paper: CDFKD-MFS: Collaborative Data-free Knowledge Distillation via Multi-level Feature Sharing. It is an extended version of the ICME21 paper 'Model Compression via Collaborative Data-Free Knowledge Distillation for Edge Intelligence'.



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

Run more implementations with configurations in the `scripts` folder.


## Citation

If you find our code useful for your research, please cite our paper.

```
@article{hao2022cdfkd,
  title={CDFKD-MFS: Collaborative Data-Free Knowledge Distillation via Multi-Level Feature Sharing},
  author={Hao, Zhiwei and Luo, Yong and Wang, Zhi and Hu, Han and An, Jianping},
  journal={IEEE Transactions on Multimedia},
  volume={24},
  pages={4262--4274},
  year={2022},
  publisher={IEEE}
}
```


## Reference

[CDFKD](https://github.com/Hao840/pytorch-CDFKD)

[CMI](https://github.com/zju-vipa/CMI)

[DFAD](https://github.com/VainF/Data-Free-Adversarial-Distillation)

[DeepInversion](https://github.com/NVlabs/DeepInversion)
