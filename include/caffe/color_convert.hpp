#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
  void convert_color(const cv::Mat bgr, cv::Mat& converted, TransformationParameter_ColorTransFormation color);
}
