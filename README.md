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
</p>

Official implementation of "Object-level Geometric Structure Preserving for Natural Image Stitching".

## Install

1. Compile ```Opencv 4.4.0```, ```VLFEAT``` and ```Eigen``` locally.

2. Create a new Visual Studio ```.sln```, and add all ```.cpp``` and ```.h``` files into this .sln.

3. Set HED file paths in ```EdgeDetection.cpp```.

## StitchBench

StitchBench is by far the most comprehensive image stitching dataset.
A sample image pair is provided in ```./input-data/AANAP-01_skyline```. 
StitchBench will be released soon.

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
