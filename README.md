<h1 align = "center">
  OBJ-GSP
</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2402.12677">
        <img alt="Paper" src="http://img.shields.io/badge/Paper-arXiv%3A2402.12677-B31B1B.svg">
    </a>
    <a href="https://huggingface.co/datasets/RussRobin/StitchBench">
        <img alt="Benchmark" src="https://img.shields.io/badge/🤗%20Benchmark-StitchBench-blue">
    </a>
    <a href="https://huggingface.co/datasets/RussRobin/Aerial234">
        <img alt="Benchmark" src="https://img.shields.io/badge/🤗%20Benchmark-Aerial234-green">
    </a>
</p>

Official implementation of "Object-level Geometric Structure Preserving for Natural Image Stitching".

## Install

1. Compile ```Opencv 4.4.0```, ```VLFEAT``` and ```Eigen``` locally.

2. Create a new Visual Studio ```.sln```, and add all ```.cpp``` and ```.h``` files into this .sln.

3. Set HED file paths in ```EdgeDetection.cpp```.

## StitchBench

StitchBench is by far the most comprehensive image stitching dataset.
A sample image pair is provided in ```./input-data/AANAP-01_skyline```. 
StitchBench will be open-sourced upon publication of our paper. 
[HuggingFace](https://huggingface.co/datasets/RussRobin/StitchBench).

## Aerial234
Aerial234 is a open-source dataset of 234 aerial images for image stitching.
We used a drone to continuously scan an area of Southeast University and collected this dataset. 
It’s quite a challenging dataset, and we’re curious if there’s a method to stitch these 234 aerial images into a single panorama.

Dataset available at: [HuggingFace](https://huggingface.co/datasets/RussRobin/Aerial234).

Our work on aerial image stiching (just a preliminary attempt): 
[UAV image stitching by estimating orthograph with RGB cameras](https://www.sciencedirect.com/science/article/pii/S1047320323000858).

## Segment Anything Model Script
Run ```.sln``` and you will find 0-original.png in the ```./``` folder.
Upload the image to Google Colab and run sam.ipynb to get SAM features and put it in ./ folder.

## Usage
For any questions, please feel free to open an issue.
```
@article{cai2024object,
  title={Object-level Geometric Structure Preserving for Natural Image Stitching},
  author={Cai, Wenxiao and Yang, Wankou},
  journal={arXiv preprint arXiv:2402.12677},
  year={2024}
}
```

```
@article{Cai2023UAVIS,
  title={UAV image stitching by estimating orthograph with RGB cameras},
  author={Wenxiao Cai and Songlin Du and Wankou Yang},
  journal={J. Vis. Commun. Image Represent.},
  year={2023},
  volume={94},
  pages={103835},
  url={https://api.semanticscholar.org/CorpusID:258424154}
}
```
