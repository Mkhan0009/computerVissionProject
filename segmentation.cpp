#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void hand_segmentation(Mat &skinImg, Rect &box, Mat &handMask)
{
  Mat imgLaplacian, sharp, bw, dist, dist_8u;
  // Create a kernel that we will use to sharpen our image
  Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
  filter2D(skinImg, imgLaplacian, CV_32F, kernel);
  skinImg.convertTo(sharp, CV_32F);
  Mat imgResult = sharp - imgLaplacian;
  // convert back to 8bits gray scale
  imgResult.convertTo(imgResult, CV_8UC3);
  imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

  // Create binary image from source image
  cvtColor(imgResult, bw, COLOR_BGR2GRAY);
  threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);

  // Perform the distance transform algorithm
  distanceTransform(bw, dist, DIST_L2, 3);
  // Normalize the distance image for range = {0.0, 1.0} so we can visualize and threshold it
  normalize(dist, dist, 0, 1.0, NORM_MINMAX);

  // Threshold to obtain the peaks, This will be the markers for the foreground objects
  threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
  // Dilate a bit the dist image
  Mat kernel1 = Mat::ones(3, 3, CV_8U);
  dilate(dist, dist, kernel1);

  // Create the CV_8U version of the distance image, It is needed for findContours()
  dist.convertTo(dist_8u, CV_8U);
  // Find total markers
  vector<vector<Point>> contours;
  findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  // Create the marker image for the watershed algorithm
  Mat markers = Mat::zeros(dist.size(), CV_32S);
  // Draw the foreground markers
  for (size_t i = 0; i < contours.size(); i++)
  {
    drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
  }
  // Draw the background marker
  circle(markers, Point(5, 5), 3, Scalar(255), -1);
  // Perform the watershed algorithm
  watershed(imgResult, markers);

  markers.convertTo(handMask, CV_8U);
  bitwise_not(handMask, handMask);

  Mat roi = Mat::zeros(handMask.size(), handMask.type());
  rectangle(roi, box, Scalar(255, 255, 255), -1);
  bitwise_and(handMask, roi, handMask);
}

void generate_colors(vector<Vec3b> &maskColors, int size)
{
  for (size_t i = 0; i < size; i++)
  {
    int b = theRNG().uniform(0, 256);
    int g = theRNG().uniform(0, 256);
    int r = theRNG().uniform(0, 256);
    maskColors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
  }
}

void skin_detection(Mat &img, Rect &box, Mat &skinImg)
{
  Mat skinInput, skinInput2, hsvInput, skinMaskHsv, ycrcbInput, skinMaskYcrcb, skinMask;

  Mat roi = Mat::zeros(img.size(), img.type());
  rectangle(roi, box, Scalar(255, 255, 255), -1);
  bitwise_and(img, roi, skinInput);

  cvtColor(skinInput, hsvInput, COLOR_BGR2HSV);
  inRange(hsvInput, Scalar(0, 15, 0), Scalar(17, 170, 255), skinMaskHsv);
  morphologyEx(skinMaskHsv, skinMaskHsv, MORPH_OPEN, Mat(3, 3, CV_8UC1, cv::Scalar(1)));

  cvtColor(skinInput, ycrcbInput, COLOR_BGR2YCrCb);
  inRange(ycrcbInput, Scalar(0, 135, 85), Scalar(255, 180, 135), skinMaskYcrcb);
  morphologyEx(skinMaskYcrcb, skinMaskYcrcb, MORPH_OPEN, Mat(3, 3, CV_8UC1, cv::Scalar(1)));

  bitwise_and(skinMaskYcrcb, skinMaskHsv, skinMask);

  medianBlur(skinMask, skinMask, 3);
  morphologyEx(skinMask, skinMask, MORPH_OPEN, Mat(4, 4, CV_8UC1, cv::Scalar(1)));

  bitwise_and(skinInput, skinInput, skinImg, skinMask);
}

void pixel_accuracy(int ith_img, Mat &predictedMask, Mat &trueMask)
{
  Mat truePositives, trueNegatives;
  bitwise_and(predictedMask, trueMask, truePositives);

  Mat inversePredictedMask, inverseTrueMask;
  bitwise_not(predictedMask, inversePredictedMask);
  bitwise_not(trueMask, inverseTrueMask);
  bitwise_and(inversePredictedMask, inverseTrueMask, trueNegatives);

  double truePositive = countNonZero(truePositives);
  double trueNegative = countNonZero(trueNegatives);

  double pixelAccuracy = (truePositive + trueNegative) / (trueMask.rows * trueMask.cols);
  cout << "Pixel accuracy of " << ith_img << "th image: " << pixelAccuracy << endl;
}
