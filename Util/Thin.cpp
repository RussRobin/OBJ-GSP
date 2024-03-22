///
/// Skeletonization code 
///

#include "Thin.h";


Mat g_srcImage, g_dstImage, g_midImage, g_grayImage, imgHSVMask;//原始图、中间图和效果图

int _size = 120;
float start_time, end_time, sum_time;


void ThinSubiteration1(Mat& pSrc, Mat& pDst);
void ThinSubiteration2(Mat& pSrc, Mat& pDst);
void normalizeLetter(Mat& inputarray, Mat& outputarray);
void Line_reflect(Mat& inputarray, Mat& outputarray);
void Delete_smallregions(Mat& pSrc, Mat& pDst);


void thin(Mat srcImage, Mat& dst, double kernalSizeTimes) {
	g_srcImage = srcImage;

	g_grayImage = g_srcImage.clone();

	threshold(g_grayImage, imgHSVMask, threshold_value, 255, THRESH_BINARY);
	int kernalSize = kernalSizeTimes * 13;

	g_midImage = Mat::zeros(imgHSVMask.size(), CV_8UC1);  //绘制
	int erosion_size = 1;
	Mat erode_element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));

	dilate(imgHSVMask, imgHSVMask, erode_element);
	erode(imgHSVMask, imgHSVMask, erode_element);

	normalizeLetter(imgHSVMask, g_dstImage);

	normalize(g_dstImage, g_midImage, 0, 255, NORM_MINMAX, CV_8U);
	g_midImage.copyTo(dst);


}

void ThinSubiteration1(Mat& pSrc, Mat& pDst) {
	int rows = pSrc.rows;
	int cols = pSrc.cols;
	pSrc.copyTo(pDst);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (pSrc.at<float>(i, j) == 1.0f) {
				int neighbor0 = (int)pSrc.at<float>(i - 1, j - 1);
				int neighbor1 = (int)pSrc.at<float>(i - 1, j);
				int neighbor2 = (int)pSrc.at<float>(i - 1, j + 1);
				int neighbor3 = (int)pSrc.at<float>(i, j + 1);
				int neighbor4 = (int)pSrc.at<float>(i + 1, j + 1);
				int neighbor5 = (int)pSrc.at<float>(i + 1, j);
				int neighbor6 = (int)pSrc.at<float>(i + 1, j - 1);
				int neighbor7 = (int)pSrc.at<float>(i, j - 1);
				int C = int(~neighbor1 & (neighbor2 | neighbor3)) +
					int(~neighbor3 & (neighbor4 | neighbor5)) +
					int(~neighbor5 & (neighbor6 | neighbor7)) +
					int(~neighbor7 & (neighbor0 | neighbor1));
				if (C == 1) {
					int N1 = int(neighbor0 | neighbor1) +
						int(neighbor2 | neighbor3) +
						int(neighbor4 | neighbor5) +
						int(neighbor6 | neighbor7);
					int N2 = int(neighbor1 | neighbor2) +
						int(neighbor3 | neighbor4) +
						int(neighbor5 | neighbor6) +
						int(neighbor7 | neighbor0);
					int N = min(N1, N2);
					if ((N == 2) || (N == 3)) {
						int c3 = (neighbor1 | neighbor2 | ~neighbor4) & neighbor3;
						if (c3 == 0) {
							pDst.at<float>(i, j) = 0.0f;
						}
					}
				}
			}
		}
	}
}

void ThinSubiteration2(Mat& pSrc, Mat& pDst) {
	int rows = pSrc.rows;
	int cols = pSrc.cols;
	pSrc.copyTo(pDst);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (pSrc.at<float>(i, j) == 1.0f) {
				int neighbor0 = (int)pSrc.at<float>(i - 1, j - 1);
				int neighbor1 = (int)pSrc.at<float>(i - 1, j);
				int neighbor2 = (int)pSrc.at<float>(i - 1, j + 1);
				int neighbor3 = (int)pSrc.at<float>(i, j + 1);
				int neighbor4 = (int)pSrc.at<float>(i + 1, j + 1);
				int neighbor5 = (int)pSrc.at<float>(i + 1, j);
				int neighbor6 = (int)pSrc.at<float>(i + 1, j - 1);
				int neighbor7 = (int)pSrc.at<float>(i, j - 1);
				int C = int(~neighbor1 & (neighbor2 | neighbor3)) +
					int(~neighbor3 & (neighbor4 | neighbor5)) +
					int(~neighbor5 & (neighbor6 | neighbor7)) +
					int(~neighbor7 & (neighbor0 | neighbor1));
				if (C == 1) {
					int N1 = int(neighbor0 | neighbor1) +
						int(neighbor2 | neighbor3) +
						int(neighbor4 | neighbor5) +
						int(neighbor6 | neighbor7);
					int N2 = int(neighbor1 | neighbor2) +
						int(neighbor3 | neighbor4) +
						int(neighbor5 | neighbor6) +
						int(neighbor7 | neighbor0);
					int N = min(N1, N2);
					if ((N == 2) || (N == 3)) {
						int E = (neighbor5 | neighbor6 | ~neighbor0) & neighbor7;
						if (E == 0) {
							pDst.at<float>(i, j) = 0.0f;
						}
					}
				}
			}
		}
	}
}

void normalizeLetter(Mat& inputarray, Mat& outputarray) {
	bool bDone = false;
	int rows = inputarray.rows;
	int cols = inputarray.cols;

	inputarray.convertTo(inputarray, CV_32FC1);

	inputarray.copyTo(outputarray);

	outputarray.convertTo(outputarray, CV_32FC1);

	Mat p_enlarged_src = Mat(rows + 2, cols + 2, CV_32FC1);
	for (int i = 0; i < (rows + 2); i++) {
		p_enlarged_src.at<float>(i, 0) = 0.0f;
		p_enlarged_src.at<float>(i, cols + 1) = 0.0f;
	}
	for (int j = 0; j < (cols + 2); j++) {
		p_enlarged_src.at<float>(0, j) = 0.0f;
		p_enlarged_src.at<float>(rows + 1, j) = 0.0f;
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (inputarray.at<float>(i, j) >= threshold_value) {
				p_enlarged_src.at<float>(i + 1, j + 1) = 1.0f;
			}
			else
				p_enlarged_src.at<float>(i + 1, j + 1) = 0.0f;
		}
	}

	Mat p_thinMat1 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
	Mat p_thinMat2 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
	Mat p_cmp = Mat::zeros(rows + 2, cols + 2, CV_8UC1);

	while (bDone != true) {
		ThinSubiteration1(p_enlarged_src, p_thinMat1);
		ThinSubiteration2(p_thinMat1, p_thinMat2);
		compare(p_enlarged_src, p_thinMat2, p_cmp, 0);
		int num_non_zero = countNonZero(p_cmp);
		if (num_non_zero == (rows + 2) * (cols + 2)) {
			bDone = true;
		}
		p_thinMat2.copyTo(p_enlarged_src);
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			outputarray.at<float>(i, j) = p_enlarged_src.at<float>(i + 1, j + 1);
		}
	}
}

void Line_reflect(Mat& inputarray, Mat& outputarray)
{
	int rows = inputarray.rows;
	int cols = inputarray.cols;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (inputarray.at<float>(i, j) == 1.0f) {
				outputarray.at<float>(i, j) = 0.0f;
			}
		}
	}
}


void Delete_smallregions(Mat& pSrc, Mat& pDst)
{

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(pSrc, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

	vector<vector<Point>>::iterator k;

	for (k = contours.begin(); k != contours.end();)
	{
		if (contourArea(*k, false) < _size)
		{
			k = contours.erase(k);
		}
		else
			++k;
	}


	for (int i = 0; i < contours.size(); i++)
	{
		for (int j = 0; j < contours[i].size(); j++)
		{
			Point P = Point(contours[i][j].x, contours[i][j].y);
		}
		drawContours(pDst, contours, i, Scalar(255), -1, 8);
	}

}

void thinTest(Mat srcImage, Mat& dst, double kernalSizeTimes) {
	g_srcImage = srcImage;

	g_grayImage = g_srcImage.clone();

	int kernalSize = kernalSizeTimes * 13;
	adaptiveThreshold(g_grayImage, imgHSVMask, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, (kernalSize % 2) == 1 ? kernalSize : kernalSize + 1, -30);
	imshow("After Binary Image", imgHSVMask);

	g_midImage = Mat::zeros(imgHSVMask.size(), CV_8UC1);  
	int erosion_size = 1; 
	Mat erode_element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	dilate(imgHSVMask, imgHSVMask, erode_element);
	erode(imgHSVMask, imgHSVMask, erode_element);

	normalizeLetter(imgHSVMask, g_dstImage);

	normalize(g_dstImage, g_midImage, 0, 255, NORM_MINMAX, CV_8U);
	g_midImage.copyTo(dst);
}
