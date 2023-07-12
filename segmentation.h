#ifndef segmentation_h
#define segmentation_h

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void hand_segmentation(Mat &skinImg, Rect &box, Mat &handMask);
void generate_colors(vector<Vec3b> &maskColors, int size);
void skin_detection(Mat &img, Rect &box, Mat &skinImg);
void pixel_accuracy(int ith_img, Mat &predictedMask, Mat &trueMask);

#endif // segmentation_h 