#include "detection.h"
#include "segmentation.h"

#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using std::filesystem::current_path;

void read_true_boxes(string file_path, vector<Rect> &trueBoxes)
{
  ifstream file(file_path);
  string line;

  if (!file.is_open())
  {
    cout << file_path << " is empty" << endl;
    return;
  }

  while (getline(file, line))
  {
    stringstream linestream(line);
    int x, y, w, h;
    linestream >> x >> y >> w >> h;
    trueBoxes.push_back(Rect(x, y, w, h));
  }
}

void configure_paths(string &cwd, string &config_path, string &weights_path, string &output_path)
{
  cwd = string(current_path());
  // remove /build from the end of the path
  cwd = cwd.substr(0, cwd.find_last_of("/"));

  config_path = cwd + "/model/custom-yolov4-detector.cfg";
  weights_path = cwd + "/model/custom-yolov4-detector_final.weights";

  // create output folders if they don't exist
  output_path = cwd + "/output";
  if (!filesystem::exists(output_path))
  {
    filesystem::create_directory(output_path);
    filesystem::create_directory(output_path + "/detection");
    filesystem::create_directory(output_path + "/segmentation");
  }
}

void read_files(string test_dataset_path, string file_id, Mat &img, vector<Rect> &trueBoxes, Mat &trueMask)
{

  string image_path = test_dataset_path + "/rgb/" + file_id + ".jpg";
  string mask_path = test_dataset_path + "/mask/" + file_id + ".png";
  string det_path = test_dataset_path + "/det/" + file_id + ".txt";
  if (!filesystem::exists(image_path) or !filesystem::exists(mask_path) or !filesystem::exists(det_path))
  {
    cout << file_id << ".jpg or " << file_id << ".png or " << file_id << ".txt doesn't exist" << endl;
    exit(-1);
  }
  img = imread(image_path);
  trueMask = imread(mask_path, IMREAD_GRAYSCALE);
  if (img.empty() or trueMask.empty())
  {
    exit(-1);
  }

  read_true_boxes(det_path, trueBoxes);
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    cout << "you didn't provide your test dataset, so default test dataset will be used" << endl;
  }

  string cwd, config_path, weights_path, output_path;
  configure_paths(cwd, config_path, weights_path, output_path);

  string test_dataset_path = (argc == 2) ? string(argv[1]) : (cwd + "/hands");
  if (!filesystem::exists(test_dataset_path))
  {
    cout << "test dataset path doesn't exist" << endl;
    return -1;
  }

  dnn::DetectionModel model = load_model(config_path, weights_path);

  for (int ith_file = 1;; ith_file++)
  {
    string file_id = (ith_file < 10) ? "0" + to_string(ith_file) : to_string(ith_file);

    Mat img, imgWithPredictedBoxes, trueMask;
    vector<Rect> predictedBoxes, trueBoxes;
    read_files(test_dataset_path, file_id, img, trueBoxes, trueMask);

    img.copyTo(imgWithPredictedBoxes);
    detect_hands_and_draw_boxes(model, predictedBoxes, imgWithPredictedBoxes);

    if (predictedBoxes.size() != trueBoxes.size())
    {
      if (predictedBoxes.size() != 0)
        sync_boxes(predictedBoxes, trueBoxes);
    }

    vector<Vec3b> maskColors;
    generate_colors(maskColors, predictedBoxes.size());
    Mat predictedMask = Mat::zeros(trueMask.size(), trueMask.type()), segmentedImage;
    img.copyTo(segmentedImage);

    cout << "IOU accuracy for " << ith_file << "th image:" << endl;
    for (size_t i = 0; i < predictedBoxes.size(); i++)
    {
      iou_accuracy(ith_file, img, predictedBoxes[i], trueBoxes[i], i + 1);

      Mat skinImg, handMask;
      skin_detection(img, predictedBoxes[i], skinImg);
      hand_segmentation(skinImg, predictedBoxes[i], handMask);
      bitwise_or(predictedMask, handMask, predictedMask);
      segmentedImage.setTo(maskColors[i], handMask > 0);
    }

    pixel_accuracy(ith_file, predictedMask, trueMask);

    imwrite(output_path + "/detection/" + file_id + ".jpg", imgWithPredictedBoxes);
    imwrite(output_path + "/segmentation/" + file_id + ".jpg", segmentedImage);
  }

  return 0;
}