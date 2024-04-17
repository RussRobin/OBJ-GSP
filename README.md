# OBJ-GSP
Official implementation of Object-level Geometric Structure Preserving for Natural Image Stitching.
[[arXiv](https://arxiv.org/abs/2402.12677)].

## Install

1. Compile Opencv 4.4.0, VLFEAT and Eigen on your machine.

2. Create a new Visual Studio .sln, and add all .cpp and .h files into this .sln.

3. Set HED file paths in EdgeDetection.cpp.

## Datasets

A sample image pair is provided in ./input-data/AANAP-01_skyline. Our proposed dataset will be released soon.

## Segment Anything Model Script
Run .sln and you will find 0-original.png in the ./ folder.
Upload the image to Google Colab and run sam.ipynb to get SAM features and put it in ./ folder.

## Citation
For any questions, please feel free to start an issue.
```
@article{cai2024object,
  title={Object-level Geometric Structure Preserving for Natural Image Stitching},
  author={Cai, Wenxiao and Yang, Wankou},
  journal={arXiv preprint arXiv:2402.12677},
  year={2024}
}
```
