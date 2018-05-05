#include <opencv2/opencv.hpp>
#define DESC_NUM_BINS 8
#include <math.h>
using namespace cv;
namespace sift_ns
{
  const int TRAIN = 1;
  //const int SIGMA = 1.6;
  const int INTERVAL = 2;
  const int DETECT = 2;
  const int n_octaves = 4;
  const int n_scales = 5;
  const int C_T = 0.03;
  const int CURVE_T = 10;
  const int NUM_BINS = 36;
  const int FEATURE_WINDOW_SIZE = 16;
  const int FV_SIZE = 128;
  const int FV_THRESH = 0.2;
  const int MAX_KERNEL_SIZE = 20;
  

  class Sift
  {
    Mat image;					//The input image
    int k;

    struct KeyPoint
    {
      float x;
      float y;
      std::vector<double> mag;
      std::vector<double> ori;
      unsigned int scale;
    };

    struct KeyDescriptor
    {
      float x;
      float y;
      std::vector<double> fv;
    };

    std::vector<KeyDescriptor> keyDesc;
    std::vector<KeyPoint> keyPoints;
    int n_keyPoints;

    std::vector<std::vector<double> > scale_sigma;
    std::vector<std::vector<Mat> > blur_images;	//To hold images after applying Gaussian
    						//kernel
    std::vector<std::vector<Mat> > dog_images;	//To hold difference of gaussian images
    std::vector<std::vector<Mat> > extrema_points;
    Mat input;
    public: Sift(char*);
    protected: void generateScaleSpace();
    protected: void localExtremaDetection();
    protected: void saveAndShowImage(char*,char*,Mat);
    protected: void assignOrientation();
    protected: void createImageDescriptor();
    protected: int kernelSize(double,double);
    protected: Mat* buildGuassianTable(int,double);
    protected: void showImage(Mat);
    public: void drawKeyPoints();

  };
};
