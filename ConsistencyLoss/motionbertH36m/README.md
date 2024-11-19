# H36M training code

## Obtain results from paper

In order to achieve the results from the paper run the training script with the config file `config/pose3d/paperh36m.yaml`

The code contained in this part of the repo is a slightly modified version of [the offical MotionBERT repo](https://github.com/Walter0807/MotionBERT).
We have changed the training a bit and added our consistency loss. Consider citing both works if you use this code.

Please see the original repo for info on how to extract the H36M dataset to the correct format.

## Citation

If you find our work useful for your project, please consider citing both our work and the original MotionBERT work:

```bibtex
@ARTICLE{ingwersen2024consistency,
       author = {Ingwersen, Christian Keilstrup and Tirsgaard, Rasmus and Nylander, Rasmus and Jensen, Janus N{\o}rtoft and Dahl, Anders Bjorholm and Hannemose, Morten Rieger},
        title = "{Two Views Are Better than One: Monocular 3D Pose Estimation with Multiview Consistency}",
      journal = {arXiv e-prints},
         year = 2024,
}
```

```bibtex
@inproceedings{ingwersen2023sportspose,
title={SportsPose: A Dynamic 3D Sports Pose Dataset},
author={Ingwersen, Christian Keilstrup and Mikkelstrup, Christian and Jensen, 
    Janus N{\o}rtoft and Hannemose, Morten Rieger and Dahl, Anders Bjorholm},
booktitle={Proceedings of the IEEE/CVF International Workshop on Computer Vision in Sports},
year={2023}
}
```

```bibtex
@inproceedings{motionbert2022,
  title     =   {MotionBERT: A Unified Perspective on Learning Human Motion Representations}, 
  author    =   {Zhu, Wentao and Ma, Xiaoxuan and Liu, Zhaoyang and Liu, Libin and Wu, Wayne and Wang, Yizhou},
  booktitle =   {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      =   {2023},
}
```
