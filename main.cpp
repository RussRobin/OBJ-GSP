

#include <iostream>
#include "./Stitching/NISwGSP_Stitching.h"
#include "./Debugger/TimeCalculator.h"

#define _CRT_SECURE_NO_WARNINGS 

using namespace std;


int GRID_SIZE_w = 40;
int GRID_SIZE_h = 40;


int main(int argc, const char* argv[]) {
	int num_data = 2; // dataset number + 1
	const char* data_list[] = { "nothing-here","AANAP-01_skyline"};

	Eigen::initParallel();
	CV_DNN_REGISTER_LAYER_CLASS(Crop, CropLayer);
	cout << "nThreads = " << Eigen::nbThreads() << endl;
	cout << "[#Images : " << num_data - 1 << "]" << endl;

	time_t start = clock();
	TimeCalculator timer;
	for (int i = 1; i < num_data; ++i) {
		cout << "i = " << i << ", [Images : " << data_list[i] << "]" << endl;
		MultiImages multi_images(data_list[i], LINES_FILTER_WIDTH, LINES_FILTER_LENGTH);

		NISwGSP_Stitching niswgsp(multi_images);
		niswgsp.setWeightToAlignmentTerm(1); 
		niswgsp.setWeightToLocalSimilarityTerm(0.75); 
		niswgsp.setWeightToGlobalSimilarityTerm(6, 20, GLOBAL_ROTATION_2D_METHOD);
		niswgsp.setWeightToContentPreservingTerm(1.5);
		Mat blend_linear;
		vector<vector<Point2> > original_vertices;
		if (RUN_TYPE != 0) {
			blend_linear = niswgsp.solve_content(BLEND_LINEAR, original_vertices);
		}
		else {
			blend_linear = niswgsp.solve(BLEND_LINEAR, original_vertices);
		}
		time_t end = clock();
		cout << "Time:" << double(end - start) / CLOCKS_PER_SEC << endl;
		niswgsp.writeImage(blend_linear, BLENDING_METHODS_NAME[BLEND_LINEAR]);

		niswgsp.assessment(original_vertices);
	}


	return 0;
}