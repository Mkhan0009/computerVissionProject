#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const vector<Scalar> boxColors = {Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0)};

dnn::DetectionModel load_model(string config_path, string weights_path)
{
  dnn::Net net = dnn::readNetFromDarknet(config_path, weights_path);
  net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
  net.setPreferableTarget(dnn::DNN_TARGET_CPU);
  dnn::DetectionModel model = dnn::DetectionModel(net);
  model.setInputParams(1. / 255, Size(416, 416), Scalar(), true);
  return model;
}

void detect_hands_and_draw_boxes(dnn::DetectionModel &model, vector<Rect> &boxes, Mat &imgWithBoxes)
{
  vector<int> classIds;
  vector<float> confidences;
  model.detect(imgWithBoxes, classIds, confidences, boxes, .2, .4);

  int detections = classIds.size();

  for (int i = 0; i < detections; ++i)
  {
    auto classId = classIds[i];
    const auto color = boxColors[classId % boxColors.size()];
    rectangle(imgWithBoxes, boxes[i], color, 3);
  }
}

void sync_boxes(vector<Rect> &predictedBoxes, vector<Rect> &trueBoxes)
{
  vector<Rect> boxes_temp;
  for (int i = 0; i < trueBoxes.size(); i++)
  {
    int min_dist_index = INT_MAX;
    double min_dist = INT_MAX;
    for (int j = 0; j < predictedBoxes.size(); j++)
    {
      double distance = sqrt(pow(predictedBoxes[j].x - trueBoxes[i].x, 2) + pow(predictedBoxes[j].y - trueBoxes[i].y, 2));
      if (distance < min_dist)
      {
        min_dist = distance;
        min_dist_index = j;
      }
    }
    boxes_temp.push_back(predictedBoxes[min_dist_index]);
  }
  predictedBoxes = boxes_temp;
}

void iou_accuracy(int a, Mat &img, Rect &predictedBox, Rect &trueBox, int ith_box)
{
  Mat predictedRoi = Mat::zeros(img.size(), img.type());
  rectangle(predictedRoi, predictedBox, Scalar(255, 255, 255), -1);

  Mat trueRoi = Mat::zeros(img.size(), img.type());
  rectangle(trueRoi, trueBox, Scalar(255, 255, 255), -1);

  Mat i, u;
  bitwise_and(predictedRoi, trueRoi, i);
  bitwise_or(predictedRoi, trueRoi, u);
  cvtColor(i, i, COLOR_BGR2GRAY);
  cvtColor(u, u, COLOR_BGR2GRAY);
  double iou = (double)countNonZero(i) / (double)countNonZero(u);
  cout << "\t" << ith_box << "th hand: " << iou << endl;
}
