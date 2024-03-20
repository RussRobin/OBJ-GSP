#pragma once
#include "EdgeDetection.h"
#include <cassert>

void edgeDetection(cv::Mat& src, cv::Mat& dst, double threshold)
{
	Mat img = src.clone();
	Size reso(img.rows, img.cols);
	Mat blob = cv::dnn::blobFromImage(img, threshold, reso, false, false);

	//Set your HED file paths.   
	assert(1 == 0); 
	string modelCfg = R"(path\to\deploy.prototxt)"; 
	string modelBin = R"(path\to\hed_pretrained_bsds.caffemodel)";
	Net net = cv::dnn::readNet(modelCfg, modelBin);
	if (net.empty()) {
		std::cout << "net empty" << std::endl;
	}
	net.setInput(blob);
	Mat out = net.forward();
	resize(out.reshape(1, reso.height), out, img.size());

	out.convertTo(dst, CV_8UC1, 255);
}
