#include "caffe/color_convert.hpp"

namespace caffe
{
  void convert_BGR_LAB(const cv::Mat bgr, cv::Mat& Lab);
  void convert_BGR_LUV(const cv::Mat bgr, cv::Mat& Luv);
  void convert_BGR_YO1O2(const cv::Mat bgr, cv::Mat& YO1O2);
  void convert_BGR_I1I2I3(const cv::Mat bgr, cv::Mat& I1I2I3);
  void convert_BGR_dRdGdB(const cv::Mat bgr, cv::Mat& dRdGdB);
  void convert_BGR_RGBdRdGdB(const cv::Mat bgr, cv::Mat& RGBdRdGdB);
  void convert_BGR_dRdGdB_internal(const cv::Mat bgrSplitted[3], cv::Mat& dR, cv::Mat& dG, cv::Mat& dB);
  void convert_BGR_HSV(const cv::Mat bgr, cv::Mat& HSV);

  void convert_BGR_YUV(const cv::Mat bgr, cv::Mat& converted);
  void convert_BGR_YIQ(const cv::Mat bgr, cv::Mat& converted);
  void convert_BGR_YPbPr(const cv::Mat bgr, cv::Mat& converted);
  void convert_BGR_YDbDr(const cv::Mat bgr, cv::Mat& converted);
  void convert_BGR_YCbCr(const cv::Mat bgr, cv::Mat& converted);

  void bgrTransformation(const cv::Mat bgr, float transfomMatrix[3][3], float alpha[3], float delta[3], cv::Mat& transfomed);

  void convert_color(const cv::Mat bgr, cv::Mat& converted, TransformationParameter_ColorTransFormation color)
  {
    switch(color)
    {
      case TransformationParameter_ColorTransFormation_LAB:
        convert_BGR_LAB(bgr, converted);
        break;
      case TransformationParameter_ColorTransFormation_LUV:
        convert_BGR_LUV(bgr, converted);
        break;
      case TransformationParameter_ColorTransFormation_YO1O2:
        convert_BGR_YO1O2(bgr, converted);
        break;
      case TransformationParameter_ColorTransFormation_I1I2I3:
        convert_BGR_I1I2I3(bgr, converted);
        break;
      case TransformationParameter_ColorTransFormation_dRdGdB:
        convert_BGR_dRdGdB(bgr, converted);
        break;
      case TransformationParameter_ColorTransFormation_RGBdRdGdB:
        convert_BGR_RGBdRdGdB(bgr, converted);
        break;
      case TransformationParameter_ColorTransFormation_HSV:
        convert_BGR_HSV(bgr, converted);
        break;


      case TransformationParameter_ColorTransFormation_YUV:
        convert_BGR_YUV(bgr, converted);
        break;
      case TransformationParameter_ColorTransFormation_YIQ:
        convert_BGR_YIQ(bgr, converted);
        break;
      case TransformationParameter_ColorTransFormation_YPbPr:
        convert_BGR_YPbPr(bgr, converted);
        break;
      case TransformationParameter_ColorTransFormation_YDbDr:
        convert_BGR_YDbDr(bgr, converted);
        break;
      case TransformationParameter_ColorTransFormation_YCbCr:
        convert_BGR_YCbCr(bgr, converted);
        break;


      case TransformationParameter_ColorTransFormation_RGB:
      default:
        bgr.convertTo(converted, CV_32F, 1.0);
        break;
    }
  }

  void convert_BGR_YUV(const cv::Mat bgr, cv::Mat& ret) {
    float transfomMatrix[3][3] = {{ 0.299,  0.587,  0.114},
                                  {-0.147, -0.289,  0.436},
                                  { 0.615, -0.515, -0.100}};
    float alpha[3] = {1.0, 1.0, 1.0};
    float delta[3] = {0.0, 0.0, 0.0};

    bgrTransformation(bgr, transfomMatrix, alpha, delta, ret);
  }

  void convert_BGR_YIQ(const cv::Mat bgr, cv::Mat& ret) {
    float transfomMatrix[3][3] = {{0.299,     0.587,     0.114   },
                                  {0.595716, -0.274453, -0.321263},
                                  {0.211456, -0.522591,  0.311135}};
    float alpha[3] = {1.0, 1.0, 1.0};
    float delta[3] = {0.0, 0.0, 0.0};

    bgrTransformation(bgr, transfomMatrix, alpha, delta, ret);
  }

  void convert_BGR_YPbPr(const cv::Mat bgr, cv::Mat& ret) {
    float transfomMatrix[3][3] = {{ 0.299,      0.587,     0.114   },
                                  {-0.1687367, -0.331264,  0.5     },
                                  { 0.5,       -0.418688, -0.081312}};
    float alpha[3] = {1.0, 1.0, 1.0};
    float delta[3] = {0.0, 0.0, 0.0};

    bgrTransformation(bgr, transfomMatrix, alpha, delta, ret);
  }

  void convert_BGR_YDbDr(const cv::Mat bgr, cv::Mat& ret) {
    float transfomMatrix[3][3] = {{ 0.299,  0.587, 0.114},
                                  {-0.450, -0.883, 1.333},
                                  {-1.333,  1.116, 0.217}};
    float alpha[3] = {1.0, 1.0, 1.0};
    float delta[3] = {0.0, 0.0, 0.0};

    bgrTransformation(bgr, transfomMatrix, alpha, delta, ret);
  }

  void convert_BGR_YCbCr(const cv::Mat bgr, cv::Mat& ret) {
    float transfomMatrix[3][3] = {{ 65.481, 128.553,  24.966 },
                                  {-37.797, -74.203, 112.0   },
                                  {112.0,   -93.786, -18.214}};
    float alpha[3] = { 1.0,   1.0,   1.0};
    float delta[3] = {16.0, 128.0, 128.0};

    bgrTransformation(bgr, transfomMatrix, alpha, delta, ret);
  }

  void bgrTransformation(const cv::Mat bgr, float transfomMatrix[3][3], float alpha[3], float delta[3], cv::Mat& transfomed) {
    std::vector<cv::Mat> array_to_merge;
    cv::Mat bgr32F;
    cv::Mat bgrSplitted[3]; // 0 - B, 1 - G, 2 - R

    bgr.convertTo(bgr32F, CV_32F, 1.0/255.0);
    cv::split(bgr32F, bgrSplitted);

    cv::Mat channel1 = alpha[0] * ( bgrSplitted[2] * transfomMatrix[0][0] + bgrSplitted[1] * transfomMatrix[0][1] + bgrSplitted[0] * transfomMatrix[0][2] ) + delta[0];
    cv::Mat channel2 = alpha[1] * (-bgrSplitted[2] * transfomMatrix[1][0] - bgrSplitted[1] * transfomMatrix[1][1] + bgrSplitted[0] * transfomMatrix[1][2] ) + delta[1];
    cv::Mat channel3 = alpha[2] * ( bgrSplitted[2] * transfomMatrix[2][0] - bgrSplitted[1] * transfomMatrix[2][1] - bgrSplitted[0] * transfomMatrix[2][2] ) + delta[2];

    array_to_merge.push_back(channel1);
    array_to_merge.push_back(channel2);
    array_to_merge.push_back(channel3);

    cv::merge(array_to_merge, transfomed);

    bgr32F.release();
    bgrSplitted[0].release();
    bgrSplitted[1].release();
    bgrSplitted[2].release();
    channel1.release();
    channel2.release();
    channel3.release();
  }

  void convert_BGR_LAB(const cv::Mat bgr, cv::Mat& Lab)
  {
    bgr.convertTo(Lab, CV_32F, 1.0/255.0);
    cv::cvtColor(Lab, Lab, CV_BGR2Lab);
  }

  void convert_BGR_LUV(const cv::Mat bgr, cv::Mat& Luv)
  {
    bgr.convertTo(Luv, CV_32F, 1.0/255.0);
    cv::cvtColor(Luv, Luv, CV_BGR2Luv);
  }

  void convert_BGR_dRdGdB_internal(const cv::Mat bgrSplitted[3], cv::Mat& dR, cv::Mat& dG, cv::Mat& dB)
  {
    int R = 2, G = 1, B = 0;  

    // (R - G) + (R - B)
    dR = (bgrSplitted[R] - bgrSplitted[G]) + (bgrSplitted[R] - bgrSplitted[B]);
    // (G - R) + (G - B)
    dG = (bgrSplitted[G] - bgrSplitted[R]) + (bgrSplitted[G] - bgrSplitted[B]);
    // (B - R) + (B - G)
    dB = (bgrSplitted[B] - bgrSplitted[R]) + (bgrSplitted[B] - bgrSplitted[G]);
  }

  void convert_BGR_HSV(const cv::Mat bgr, cv::Mat& HSV)
  {
    std::vector<cv::Mat> array_to_merge;
    cv::Mat bgr32F;
    cv::Mat hsvSplitted[3]; // 0 - H, 1 - S, 2 - V
    cv::Mat hsv; // temp

    bgr.convertTo(bgr32F, CV_32F, 1.0/255.0);
    cv::cvtColor(bgr32F, hsv, CV_BGR2HSV);

    cv::split(hsv, hsvSplitted);

    // 0 <= H <= 2pi
    hsvSplitted[0] /= 180.0;
    hsvSplitted[0] *= CV_PI;

    array_to_merge.push_back(hsvSplitted[0]); // H
    array_to_merge.push_back(hsvSplitted[1]); // S
    array_to_merge.push_back(hsvSplitted[2]); // V

    cv::merge(array_to_merge, HSV);

    bgr32F.release();
    hsv.release();
    hsvSplitted[0].release();
    hsvSplitted[1].release();
    hsvSplitted[2].release();
  }

  void convert_BGR_dRdGdB(const cv::Mat bgr, cv::Mat& dRdGdB)
  {
    std::vector<cv::Mat> array_to_merge;
    cv::Mat bgr32F;
    cv::Mat bgrSplitted[3]; // 0 - B, 1 - G, 2 - R

    bgr.convertTo(bgr32F, CV_32F);
    cv::split(bgr32F, bgrSplitted);

    cv::Mat dR, dG, dB;
    
    convert_BGR_dRdGdB_internal(bgrSplitted, dR, dG, dB);

    array_to_merge.push_back(dR);
    array_to_merge.push_back(dG);
    array_to_merge.push_back(dB);

    cv::merge(array_to_merge, dRdGdB);

    bgr32F.release();
    bgrSplitted[0].release();
    bgrSplitted[1].release();
    bgrSplitted[2].release();
    dR.release();
    dG.release();
    dB.release();
  }

  void convert_BGR_RGBdRdGdB(const cv::Mat bgr, cv::Mat& RGBdRdGdB)
  {
    std::vector<cv::Mat> array_to_merge;
    cv::Mat bgr32F;
    cv::Mat bgrSplitted[3]; // 0 - B, 1 - G, 2 - R

    int R = 2;
    int G = 1;
    int B = 0;

    bgr.convertTo(bgr32F, CV_32F);
    cv::split(bgr32F, bgrSplitted);

    cv::Mat dR, dG, dB;
    
    convert_BGR_dRdGdB_internal(bgrSplitted, dR, dG, dB);

    array_to_merge.push_back(bgrSplitted[R]);
    array_to_merge.push_back(bgrSplitted[G]);
    array_to_merge.push_back(bgrSplitted[B]);

    array_to_merge.push_back(dR);
    array_to_merge.push_back(dG);
    array_to_merge.push_back(dB);

    cv::merge(array_to_merge, RGBdRdGdB);

    bgr32F.release();
    bgrSplitted[0].release();
    bgrSplitted[1].release();
    bgrSplitted[2].release();
    dR.release();
    dG.release();
    dB.release();
  }

  void convert_BGR_I1I2I3(const cv::Mat bgr, cv::Mat& I1I2I3)
  {
    std::vector<cv::Mat> array_to_merge;
    cv::Mat bgr32F;
    cv::Mat bgrSplitted[3]; // 0 - B, 1 - G, 2 - R

    bgr.convertTo(bgr32F, CV_32F);
    cv::split(bgr32F, bgrSplitted);

    cv::Mat I1 = (bgrSplitted[2] + bgrSplitted[1] + bgrSplitted[0]) / 3.0;
    cv::Mat I2 = (bgrSplitted[2] - bgrSplitted[0]) / 2.0;
    cv::Mat I3 = (2*bgrSplitted[1] - bgrSplitted[2] - bgrSplitted[0]) / 4.0;

    array_to_merge.push_back(I1);
    array_to_merge.push_back(I2);
    array_to_merge.push_back(I3);

    cv::merge(array_to_merge, I1I2I3);

    bgr32F.release();
    bgrSplitted[0].release();
    bgrSplitted[1].release();
    bgrSplitted[2].release();
    I1.release();
    I2.release();
    I3.release();
  }

  void convert_BGR_YO1O2(const cv::Mat bgr, cv::Mat& YO1O2)
  {
    std::vector<cv::Mat> array_to_merge;
    cv::Mat bgr32F;
    cv::Mat bgrSplitted[3]; // 0 - B, 1 - G, 2 - R

    bgr.convertTo(bgr32F, CV_32F);
    cv::split(bgr32F, bgrSplitted);

    int R = 2, G = 1, B = 0;

    cv::Mat Y = (0.2857 * bgrSplitted[R]) + (0.5714 * bgrSplitted[G]) + (0.1429 * bgrSplitted[B]);
    cv::Mat O1 = bgrSplitted[R] - bgrSplitted[G];
    cv::Mat O2 = (2.0 * bgrSplitted[B]) - bgrSplitted[R] - bgrSplitted[G];

    array_to_merge.push_back(Y);
    array_to_merge.push_back(O1);
    array_to_merge.push_back(O2);

    cv::merge(array_to_merge, YO1O2);

    bgr32F.release();
    bgrSplitted[0].release();
    bgrSplitted[1].release();
    bgrSplitted[2].release();
    Y.release();
    O1.release();
    O2.release();
  }

}

