#ifndef detection_h
#define detection_h

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

dnn::DetectionModel load_model(string config_path, string weights_path);
void detect_hands_and_draw_boxes(dnn::DetectionModel &model, vector<Rect> &boxes, Mat &imgWithBoxes);
void iou_accuracy(int a, Mat &img, Rect &predictedBox, Rect &trueBox, int ith_box);
void sync_boxes(vector<Rect> &predictedBoxes, vector<Rect> &trueBoxes);

#endif // detection_h