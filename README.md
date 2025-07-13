<h1 align = "center">
OBJ-GSP (AAAI 2025)
</h1>

<p align="center">
    <a href="https://ojs.aaai.org/index.php/AAAI/article/view/32188">
        <img alt="AAAI Paper" src="http://img.shields.io/badge/Paper-AAAI-B31B1B.svg">
    </a>
    <a href="https://arxiv.org/abs/2402.12677">
        <img alt="arXiv" src="http://img.shields.io/badge/Full Length Paper-arXiv-orange">
    </a>
    <a href="https://huggingface.co/datasets/RussRobin/StitchBench">
        <img alt="Benchmark" src="https://img.shields.io/badge/🤗%20Benchmark-StitchBench-green">
    </a>
    <a href="https://huggingface.co/datasets/RussRobin/Aerial234">
        <img alt="Benchmark" src="https://img.shields.io/badge/🤗%20Benchmark-Aerial234-blue">
    </a>
</p>


Official implementation of *AAAI 2025* paper "Object-level Geometric Structure Preserving for Natural Image Stitching".

## Install

1. Compile ```Opencv 4.4.0```, ```VLFEAT``` and ```Eigen``` locally.

2. Create a new Visual Studio ```.sln```, and add all ```.cpp``` and ```.h``` files into this .sln.

3. Set HED file paths in ```EdgeDetection.cpp```.

## StitchBench

StitchBench is by far the most comprehensive image stitching dataset.
A sample image pair is provided in ```./input-data/AANAP-01_skyline```. 
StitchBench is available at: 
[HuggingFace](https://huggingface.co/datasets/RussRobin/StitchBench). You will be automatically granted access to it.

## Aerial234
Aerial234 is a open-source dataset of 234 aerial images for image stitching.
We used a drone to continuously scan an area of Southeast University and collected this dataset. 
It’s quite a challenging dataset, and we’re curious if there’s a method to stitch these 234 aerial images into a single panorama.

Aerial234 is available at: [HuggingFace](https://huggingface.co/datasets/RussRobin/Aerial234).

Our work on aerial image stiching (just a preliminary attempt): 
[UAV image stitching by estimating orthograph with RGB cameras](https://www.sciencedirect.com/science/article/pii/S1047320323000858).

## Segment Anything Model Script
Run ```.sln``` and you will find 0-original.png in the ```./``` folder.
Upload the image to Google Colab and run sam.ipynb to get SAM features and put it in ./ folder.

## Usage
For any questions, please feel free to open an issue.
```
@inproceedings{cai2025object,
  title={Object-level geometric structure preserving for natural image stitching},
  author={Cai, Wenxiao and Yang, Wankou},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={2},
  pages={1926--1934},
  year={2025}
}
```

```
@article{cai2023uav,
  title={UAV image stitching by estimating orthograph with RGB cameras},
  author={Cai, Wenxiao and Du, Songlin and Yang, Wankou},
  journal={Journal of Visual Communication and Image Representation},
  volume={94},
  pages={103835},
  year={2023},
  publisher={Elsevier}
}
```

We appreciate AAAI for providing Student Scholarship for this paper!
