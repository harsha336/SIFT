#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <sift.hh>
#include <string.h>

using namespace sift_ns;
using namespace cv;

Sift::Sift(char* image_name)
{
  input = imread(image_name,IMREAD_COLOR);

  if(!input.data)
  {
    std::cout << "No image!";
    return;
  }

  //printf("Image is loaded\n");

  showImage(input);

  scale_sigma = std::vector<std::vector<double> >(n_octaves);
  blur_images = std::vector<std::vector<Mat> >(n_octaves);
  dog_images = std::vector<std::vector<Mat> >(n_octaves);
  extrema_points = std::vector<std::vector<Mat> >(n_octaves);
  for (int i=0;i<n_octaves;i++)
  {
    scale_sigma[i] = std::vector<double>(n_scales);
    blur_images[i] = std::vector<Mat>(n_scales);
    dog_images[i] = std::vector<Mat>(n_scales-1);
    extrema_points[i] = std::vector<Mat>(n_scales-3);
//printf("Full init extrema[%d]:[%d]\n",i,extrema_points[i].size());
//printf("Full init dog[%d]:[%d]\n",i,dog_images[i].size());
  }

  //printf("Matrices are loaded\n");



  //Pre-process the image
  Mat gray_image;
  Mat tmp_img;
  cvtColor(input,gray_image,CV_BGR2GRAY);
  //printf("Gray image!\n");
  showImage(gray_image);

  //printf("Image type is %d",gray_image.type());

  gray_image.convertTo(gray_image,CV_32FC1,1.0/255.0);
  //printf("Converted image\n");
  //showImage(gray_image);

/*  for(int i=0;i<gray_image.cols;i++)
  {
    for(int j=0;j<gray_image.rows;j++)
    {
      printf("Image value at loacation: [%d][%d] = [%f]\n",gray_image.at<double>(j,i));
    }
  }
*/
  //printf("Normalized!\n");
  //showImage(gray_image);

  GaussianBlur(gray_image,tmp_img,Size(0,0),0.5);
  //printf("Blurred image\n");
  //showImage(tmp_img);

  pyrUp(tmp_img,image,Size(tmp_img.cols*2,tmp_img.rows*2));
  //printf("Scaled up!\n");
  GaussianBlur(image,image,Size(0,0),1.0);
  
  //Add the image in the blur_images list and sigma
  blur_images[0][0] = image;
  //k = pow(2.0f,1/INTERVAL);


  generateScaleSpace();
  localExtremaDetection();
  //printf("Types of Mat are:\nblur images:[%d]\ndog images:[%d]\nextrema points:[%d]\n",blur_images[0][0].type(),dog_images[0][0].type(),extrema_points[0][0].type());
  assignOrientation();
  createImageDescriptor();

}

void Sift::showImage(Mat im)
{
  char window[] = "SIFT";
  imshow(window,im);
  waitKey(0);
}

void Sift::generateScaleSpace()
{

  double SIGMA = sqrt(2);
  
  scale_sigma[0][0] = SIGMA*0.5;
  
  //printf("Showing image inside Sift::generateScaleSpace\n");
  showImage(blur_images[0][0]);
  //printf("Sigma at [0,0]:[%f]\n",scale_sigma[0][0]);
  for (int i=0;i<n_octaves;i++)
  {
    double sigma = SIGMA;

    for (int j=1;j<n_scales;j++)
    {
      //printf("generateScaleSpace(): Iterating through: <Octaves:[%d],Scales:[%d]>\n",i,j);
      double sigma_f = sqrt(pow(2.0,2.0/INTERVAL)-1)*sigma;
      sigma = pow(2.0,1.0/INTERVAL)*sigma;
      scale_sigma[i][j] = sigma*0.5*pow(2.0f,(float)i);
      //printf("Sigma at [%d,%d]:[%f]\n",i,j,scale_sigma[i][j]);

      GaussianBlur(blur_images[i][j-1],blur_images[i][j],Size(0,0),sigma);
      showImage(blur_images[i][j]);
      //GaussianBlur(blur_images[i][j-1],blur_images[i][j],Size(0,0),sigma_f);

      //scale_sigma[i][j] = sigma;
      //sigma = k*sigma;

      subtract(blur_images[i][j-1],blur_images[i][j],dog_images[i][j-1]);
      showImage(dog_images[i][j-1]);
    }
    if (i<n_octaves-1)
    {
      pyrDown(blur_images[i][0],blur_images[i+1][0],Size(blur_images[i][0].cols/2,blur_images[i][0].rows/2));
      scale_sigma[i+1][0] = scale_sigma[i][INTERVAL];
      //printf("Sigma at [%d,%d]:[%f]\n",i+1,0,scale_sigma[i+1][0]);
      showImage(blur_images[i+1][0]);
    }
  }

}

void Sift::localExtremaDetection()
{
  Mat_<double> prev,next,cur;
  Mat cur_img;
  char* imgname;
  int num_keyp = 0;
  int num_rem=0;
  double dxx,dyy,dxy,t_H,d_H,c_r,c_t;
  c_t = (CURVE_T+1)*(CURVE_T+1)/CURVE_T;
  for (int oi=0;oi<n_octaves;oi++)
  {
    for (int oj=1;oj<n_scales-2;oj++)
    {
      extrema_points[oi][oj-1] = Mat::zeros(dog_images[oi][oj-1].size(),CV_8UC1);
//extrema_points[oi][oj-1].setTo(Scalar(0));
      prev = dog_images[oi][oj-1];next = dog_images[oi][oj+1];cur = dog_images[oi][oj];cur_img = dog_images[oi][oj];
      //imgname = "extrema_"+oi+"_"+oj;
//printf("oi:[%d],oj:[%d]\n",oi,oj);
      
      
      for (int i=1;i<dog_images[oi][oj-1].rows-1;i++)
      {
        for (int j=1;j<dog_images[oi][oj-1].cols-1;j++)
	{
	  if(i-1<0 || j-1 < 0 || i+1>dog_images[oi][oj-1].rows-1 || j+1 > cur.cols-1)
	  {
	    printf("Maybe something wrong here!\n");
	    continue;
	  }
	  if(cur(i,j)<cur(i,j-1) && 
	      cur(i,j)<cur(i,j+1) &&
	      cur(i,j)<cur(i-1,j) && 
	      cur(i,j)<cur(i+1,j) && 
	      cur(i,j)<cur(i-1,j-1) &&
	      cur(i,j)<cur(i+1,j-1) && 
	      cur(i,j)<cur(i+1,j+1) && 
	      cur(i,j)<cur(i-1,j+1) &&
	      cur(i,j)<prev(i,j) && 
	      cur(i,j)<prev(i,j-1) && 
	      cur(i,j)<prev(i,j+1) &&
	      cur(i,j)<prev(i-1,j) && 
	      cur(i,j)<prev(i-1,j-1) && 
	      cur(i,j)<prev(i-1,j+1) &&
	      cur(i,j)<prev(i+1,j) && 
	      cur(i,j)<prev(i+1,j-1) && 
	      cur(i,j)<prev(i+1,j+1) &&
	      cur(i,j)<next(i,j) && 
	      cur(i,j)<next(i,j-1) && 
	      cur(i,j)<next(i,j+1) &&
	      cur(i,j)<next(i-1,j) && 
	      cur(i,j)<next(i-1,j-1) && 
	      cur(i,j)<next(i-1,j+1) &&
	      cur(i,j)<next(i+1,j) && 
	      cur(i,j)<next(i+1,j-1) && 
	      cur(i,j)<next(i+1,j+1))
	   {
	    extrema_points[oi][oj-1].at<uchar>(i,j) = 1;
	    num_keyp++;
	    circle(cur_img,Point(j,i),2,Scalar(255));
	   }

	   else if(cur(i,j)>cur(i,j-1) && 
	      cur(i,j)>cur(i,j+1) &&
              cur(i,j)>cur(i-1,j-1) && 
	      cur(i,j)>cur(i,j-1) && 
	      cur(i,j)>cur(i-1,j+1) &&
              cur(i,j)>cur(i+1,j-1) && 
	      cur(i,j)>cur(i+1,j) && 
	      cur(i,j)>cur(i+1,j+1) &&
              cur(i,j)>prev(i,j) && 
	      cur(i,j)>prev(i,j-1) && 
	      cur(i,j)>prev(i,j+1) &&
              cur(i,j)>prev(i-1,j) && 
	      cur(i,j)>prev(i-1,j-1) && 
	      cur(i,j)>prev(i-1,j+1) &&
              cur(i,j)>prev(i+1,j) && 
	      cur(i,j)>prev(i+1,j-1) && 
	      cur(i,j)>prev(i+1,j+1) &&
              cur(i,j)>next(i,j) && 
	      cur(i,j)>next(i,j-1) && 
	      cur(i,j)>next(i,j+1) &&
              cur(i,j)>next(i-1,j) && 
	      cur(i,j)>next(i-1,j-1) && 
	      cur(i,j)>next(i-1,j+1) &&
              cur(i,j)>next(i+1,j) && 
	      cur(i,j)>next(i+1,j-1) && 
	      cur(i,j)>next(i+1,j+1))
	   {
	     extrema_points[oi][oj-1].at<uchar>(i,j) = 1;
	     num_keyp++;
	     circle(cur_img,Point(j,i),2,Scalar(255));

	   }

	   if((extrema_points[oi][oj-1].at<uchar>(i,j) == 1) && fabs(cur(i,j)) < C_T)
	   {
	     extrema_points[oi][oj-1].at<uchar>(i,j) = 0;
	     num_rem++;
	     num_keyp--;
	     circle(cur_img,Point(j,i),2,Scalar(255));
	   }

	   if(extrema_points[oi][oj-1].at<uchar>(i,j) == 1)
	   {
	     
	       dxx = cur(i,j-1)+cur(i,j+1) - (2*cur(i,j));
	       dyy = cur(i-1,j)+cur(i+1,j) - (2*cur(i,j));
	       dxy = (cur(i-1,j-1) + cur(i+1,j+1) - cur(i+1,j-1) - cur(i-1,j+1))/4.0;
	       t_H = dxx + dyy;
	       d_H = dxx*dyy - dxy*dxy;

	       c_r = (t_H*t_H)/d_H;
	       //printf("Determinant of H: <%lf>, Trace of H: <%lf>,curvature ratio :<%f> dxx:<%f>,dyy:<%f>,dxy:<%f>\n",d_H,t_H,c_r,dxx,dyy,dxy);
	       
	       if(d_H<0 || c_r>c_t)
	       {
	         extrema_points[oi][oj-1].at<uchar>(i,j) = 0;
		 num_rem++;
		 num_keyp--;
		 circle(cur_img,Point(j,i),2,Scalar(255));
	       }
	       
	   }
	   

	}
      }
      showImage(cur_img);

    }
//printf("Extrema init size[%d]:[%d]\n",oi,extrema_points[oi].size());
  }
  printf("Number of keypoints added: <%d> removed : <%d>\n",num_keyp,num_rem);
  n_keyPoints = num_keyp;
  int count=0;

  for(int i=0;i<n_octaves;i++)
  {
    for(int j=0;j<INTERVAL;j++)
    {
      
      for(int ii=0;ii<extrema_points[i][j].rows;ii++)
      {
        for(int jj=0;jj<extrema_points[i][j].cols;jj++)
	{
	  if(extrema_points[i][j].at<uchar>(ii,jj) != 0)
	    count++;
	}
      }
    }
  }
  //printf("Local extrema computed count : <%d>",count);
 
}

void Sift::saveAndShowImage(char *imagename,char *winname,Mat image)
{
  char *imagpath = "../../images/"; //+ imagename;
  imwrite(imagpath,image);
  namedWindow(winname,CV_WINDOW_AUTOSIZE);
  imshow(winname,image);
}

void Sift::assignOrientation()
{
  //printf("Inside assignOrientation\n");
  std::vector<std::vector<Mat> > mag(n_octaves);
  std::vector<std::vector<Mat> > orient(n_octaves);
  for(int i=0;i<n_octaves;i++)
  {
    mag[i] = std::vector<Mat>(INTERVAL);
    orient[i] = std::vector<Mat>(INTERVAL);
    //printf("Mag[%d] size: [%d],Orient[%d] size: [%d]\n",i,mag[i].size(),i,orient[i].size());
  }

  //printf("magnitude and orientation vectors initialized successfully\n");


  for (int i=0;i<n_octaves;i++)
  {
    for (int j=1;j<INTERVAL+1;j++)
    {
      if(i<0 || j-1<0 || i>n_octaves-1 || j-1>n_scales-3)
      {
        //printf("Def something wrong here[%d][%d]!\n",i,j-1);
      }

      mag[i][j-1] = Mat::zeros(blur_images[i][j].size(),blur_images[i][j].type());
      orient[i][j-1] = Mat::zeros(blur_images[i][j].size(),CV_64FC1);

      //printf("Mag[%d][%d] : [%d][%d], Orient[%d][%d] : [%d][%d]\n",i,j-1,mag[i][j-1].rows,mag[i][j-1].cols,i,j-1,orient[i][j-1].rows,orient[i][j-1].cols);
      //tmp = blur_images[i][j];
      //printf("Tmp size:[%d][%d]\n",blur_images[i][j].rows,blur_images[i][j].cols);

      for (int ii=1;ii<blur_images[i][j].rows-1;ii++)
      {
        for (int ij=1;ij<blur_images[i][j].cols-1;ij++)
	{
	  if(ii+1>blur_images[i][j].rows-1 || ij > blur_images[i][j].cols-1 || ii-1 < 0 || ij-1 < 0)
	  {
	    //printf("Second maybe something wrong here!\n");
	    continue;
	  }
	  float dx = blur_images[i][j].at<float>(ii+1,ij) - blur_images[i][j].at<float>(ii-1,ij);
	  float dy = blur_images[i][j].at<float>(ii,ij+1) - blur_images[i][j].at<float>(ii,ij-1);

	  mag[i][j-1].at<float>(ii,ij) = sqrt(dx*dx + dy*dy);
	  orient[i][j-1].at<double>(ii,ij) = (atan2(dy,dx)==M_PI)? -M_PI:atan2(dy,dx);

	  if(orient[i][j-1].at<double>(ii,ij)>2*M_PI || orient[i][j-1].at<double>(ii,ij)<-2*M_PI)
	    printf("Something wrong here:<%f>",orient[i][j-1].at<double>(ii,ij));
	  //printf("Magnitude [%d][%d] at<%d,%d>:<%f>",i,j-1,ij,ii,mag[i][j-1].at<double>(ij,ii));

	}

	for(int padi=0;padi<mag[i][j-1].rows;padi++)
	{
	  mag[i][j-1].at<float>(padi,0)                          = 0.0; 
          mag[i][j-1].at<float>(padi,mag[i][j-1].cols-1)   = 0.0; 
          orient[i][j-1].at<double>(padi,0)                        = 0.0; 
          orient[i][j-1].at<double>(padi,mag[i][j-1].cols-1) = 0.0; 
	}
	for (int padj=0;padj<mag[i][j-1].cols;padj++) 
        { 
          mag[i][j-1].at<float>(0,padj)                          = 0.0; 
          mag[i][j-1].at<float>(mag[i][j-1].rows-1,padj)   = 0.0; 
          orient[i][j-1].at<double>(0,padj)                        = 0.0; 
          orient[i][j-1].at<double>(mag[i][j-1].rows-1,padj) = 0.0;
	}


      }

    }
  }


  std::vector<double> orient_hist(36);

for(int e=0;e<n_octaves;e++)
{
//printf("Number in each [%d]:[%d]\n",e,extrema_points[e].size());
}  

  Mat weightImg;
  Mat_<int> epi;
  int count=0;

  for(unsigned int i=0;i<n_octaves;i++)
  {
    int scale = pow(2.0,i);
    int row = blur_images[i][0].rows;
    int col = blur_images[i][0].cols;
    Size size = blur_images[i][0].size();
    int type = blur_images[i][0].type();

    for(unsigned int j=1;j<n_scales-2;j++)
    {
      //printf("Current sigma is <%f>\n",scale_sigma[i][j]);
      double cur_sigma = scale_sigma[i][j];
      
      //Mat *orientImg = new Mat::zeros(size,type);
      if(i<0 || i>n_octaves-1 || j<0 || j-1>INTERVAL-1)
      {
        printf("Third maybe something is wrong!\n");
      }

      GaussianBlur(mag[i][j-1],weightImg,Size(0,0),1.5*cur_sigma);
      

      unsigned int blur_size = kernelSize(1.5*cur_sigma,0.001)/2;
      //printf("Kernel size returned: <%d>\n",blur_size);

     // printf("Width,height of extrema = [%d,%d]\n",extrema_points[i][j-1].cols,extrema_points[i][j-1].rows);


      for(unsigned int ii=0;ii<row;ii++)
      {
        for(unsigned int ij=0;ij<col;ij++)
	{
	  //epi = extrema_points[i][j-1];

	  if(ii<0 || ij<0 || ii>row-1 || ij>col-1)
	  {
	    printf("Fourth maybe something is wrong here!");
	    continue;
	  }
	  if(extrema_points[i][j-1].at<uchar>(ii,ij) != 0)
	  {
	    count++;
	    //printf("Accessing ext points [%d,%d][%d,%d]:<%f>",i,j,ij,ii,extrema_points[i][j].at<double>(ij,ii));
	    
	    for(int h=0;h<36;h++)
	      orient_hist[h] = 0.0;
	    
	    for(int bi=-blur_size;bi<=static_cast<int>(blur_size);bi++)
	    {
	      for(int bj=-blur_size;bj<=static_cast<int>(blur_size);bj++)
	      {
	      
	        if(static_cast<int>(ii)+bi<0 || static_cast<int>(ij)+bj >col-1 || static_cast<int>(ij)+bj<0 || static_cast<int>(ii)+bi>row-1)
		{
		  //printf("Fifth maybe something is wrong here[%d,%d]:[%d][%d]!\n",i,j-1,ii+bi,ij+bj);
		  continue;
		}

		double cur_orient = orient[i][j-1].at<double>(ii+bi,ij+bj);
		if(!(cur_orient>=-M_PI && cur_orient<M_PI))
		  printf("Current orient = %f\n",cur_orient);
		assert(cur_orient>=-M_PI-0.002 && cur_orient<=M_PI+0.002);
		
		cur_orient += M_PI;
		

		unsigned cur_orient_deg = static_cast<unsigned int>(cur_orient * 180 / M_PI);
		assert(cur_orient_deg>=0 && cur_orient_deg<=360);
		orient_hist[cur_orient_deg/(360/10)] += weightImg.at<float>(ii+bi,ij+bj);
		
	      }
	    }
	    
	    

	    double peak = orient_hist[0];
	    int peak_ind = 0;
	    for (int p=1;p<36;p++)
	    {
	      if(orient_hist[p]>peak)
	      {
	        peak = orient_hist[p];
		peak_ind = p;
	      }
	    }

	    std::vector<double> magn;
	    std::vector<double> orientation;
	    
	    
	    for(int p=0;p<36;p++)
	    {
	      if(orient_hist[p]>=0.8*peak)
	      {
	        double x1 = p-1;
		double y1;
		double x2 = p;
		double y2 = orient_hist[p];
		double x3 = p+1;
		double y3;

		if(p == 0)
		{
		  y1 = orient_hist[9];
		  y3 = orient_hist[1];
		}
		else if(p == 9)
		{
		  y1 = orient_hist[9];
		  y3 = orient_hist[0];
		}
		else
		{
		  y1 = orient_hist[p-1];
		  y3 = orient_hist[p+1];
		}
		
		Mat par_X = (Mat_<double>(3,3) << x1*x1,x1,1,x2*x2,x2,1,x3*x3,x3,1);
		Mat par_inv_X = par_X.inv();
		double b[3];

		b[0] = par_inv_X.at<double>(0,0)*y1 + par_inv_X.at<double>(1,0)*y2 + par_inv_X.at<double>(2,0)*y3;
		b[1] = par_inv_X.at<double>(0,1)*y1 + par_inv_X.at<double>(1,1)*y2 + par_inv_X.at<double>(2,1)*y3;
		b[2] = par_inv_X.at<double>(0,2)*y1 + par_inv_X.at<double>(1,2)*y2 + par_inv_X.at<double>(2,2)*y3;

		double x0 = -b[1]/(2*b[0]);

		if(fabs(x0)>20)
		  x0 = x2;

		while(x0<0)
		  x0 += 10;

		while(x0>=10)
		  x0 -= 10;

		double x0_rad = x0 * M_PI / 180;
		assert(x0_rad >=0 && x0_rad <2*M_PI);
		x0_rad -= M_PI;
		assert(x0_rad>=-M_PI-0.002 && x0_rad<M_PI+0.002);
		orientation.push_back(x0_rad);
		magn.push_back(orient_hist[p]);
	      }
	    }
	    struct KeyPoint temp;
	    temp.x = ii*scale/2;temp.y = ij*scale/2;temp.mag = magn;temp.ori = orientation;temp.scale = i*INTERVAL+j-1;
	    keyPoints.push_back(temp);
	  }
	}
      }
    }
  }

  assert(keyPoints.size() == n_keyPoints);

}

void Sift::createImageDescriptor()
{
  std::vector<std::vector<Mat> > ipMag = std::vector<std::vector<Mat> >(n_octaves);
  std::vector<std::vector<Mat> > ipOrient = std::vector<std::vector<Mat> >(n_octaves);

  for(int i=0;i<n_octaves;i++)
  {
    ipMag[i] = std::vector<Mat>(INTERVAL);
    ipOrient[i] = std::vector<Mat>(INTERVAL);
  }

  for(int i=0;i<n_octaves;i++)
  {
    for(int j=1;j<INTERVAL+1;j++)
    {
      unsigned int row = blur_images[i][j].rows;
      unsigned int col = blur_images[i][j].cols;
      Size size = blur_images[i][j].size();
      int type = blur_images[i][j].type();

      Mat tempImg;
      //pyrUp(blur_images,tempImg,Size(blur_images[i][j].cols*2,blur_images[i][j].rows*2));

      ipMag[i][j-1] = Mat::zeros(Size(row+1,col+1),type);
      ipOrient[i][j-1] = Mat::zeros(Size(row+1,col+1),CV_64FC1);

      for(float ii=1.5;ii<row-1.5;ii++)
      {
        for(float jj=1.5;jj<col-1.5;jj++)
	{
	  
	  double dx = blur_images[i][j].at<float>(ii+1.0,jj) - blur_images[i][j].at<float>(ii-1.0,jj);
	  double dy = blur_images[i][j].at<float>(ii,jj+1.0) - blur_images[i][j].at<float>(ii,jj-1.0);

	  unsigned int iii = static_cast<unsigned int>(ii+1.0);
	  unsigned int jjj = static_cast<unsigned int>(ii+1.0);

	  assert(iii<=row && jjj<=col);

	  ipMag[i][j-1].at<float>(iii,jjj) = sqrt(dx*dx + dy*dy);
	  ipOrient[i][j-1].at<double>(iii,jjj) = (atan2(dy,dx)==M_PI)? -M_PI:atan2(dy,dx);
	}
      }

      for(unsigned int iii=0;iii<row+1;iii++)
      {
        ipMag[i][j-1].at<float>(iii,0) = 0;
	ipMag[i][j-1].at<float>(iii,col) = 0;
	ipOrient[i][j-1].at<double>(iii,0) = 0;
	ipOrient[i][j-1].at<double>(iii,col) = 0;
      }

      for(unsigned int jjj=0;jjj<col+1;jjj++)
      {
        ipMag[i][j-1].at<float>(0,jjj) = 0;
	ipMag[i][j-1].at<float>(row,jjj) = 0;
	ipOrient[i][j-1].at<double>(0,jjj) = 0;
	ipOrient[i][j-1].at<double>(row,jjj) = 0;
      }

    }
  }

  Mat *GT = buildGuassianTable(FEATURE_WINDOW_SIZE,0.5*FEATURE_WINDOW_SIZE);
  std::vector<float> hist(DESC_NUM_BINS);

  for(int k=0;k<n_keyPoints;k++)
  {
  
    unsigned int scale = keyPoints[k].scale;
    float kpx = keyPoints[k].x;
    float kpy = keyPoints[k].y;

    float descx = kpx;
    float descy = kpy;

    unsigned int ii = static_cast<unsigned int>(kpx*2)/static_cast<unsigned int>(pow(2.0,static_cast<float>(scale/n_scales)));
    unsigned int jj = static_cast<unsigned int>(kpy*2)/static_cast<unsigned int>(pow(2.0,static_cast<float>(scale/n_scales)));

    int col = blur_images[scale/n_scales][0].cols;
    int row = blur_images[scale/n_scales][0].rows;

    std::vector<double> mag = keyPoints[k].mag;
    std::vector<double> orient = keyPoints[k].ori;

    double main_mag = mag[0];
    double main_orien = orient[0];

    for(int oc=1;oc<mag.size();oc++)
    {
      if(mag[oc]>main_mag)
      {
        main_mag = mag[oc];
	main_orien = orient[oc];
      }
    }

    int hfsz = FEATURE_WINDOW_SIZE/2;
    Mat weight = Mat(Size(FEATURE_WINDOW_SIZE,FEATURE_WINDOW_SIZE),CV_32FC1);
    std::vector<double> fv(FV_SIZE);
    
    //printf("Scale:%d,Interval:%d,[%d,%d]\n",scale,INTERVAL,scale/INTERVAL,scale%INTERVAL);
    //printf("Size of ipmag:[%d,%d],Size of GT:[%d,%d],Size of weight:[%d,%d]\n",ipMag[scale/INTERVAL][scale%INTERVAL].rows,ipMag[scale/INTERVAL][scale%INTERVAL].cols,GT->rows,GT->cols,weight.rows,weight.cols);
    //printf("Size of weight:[%d][%d]\nSize of GT:[%d,%d]\nSize of ipMag:[%d,%d]\n",weight.rows,weight.cols,GT.rows,GT.cols,ipMag.rows,ipMag.cols);

    for(int fi=0;fi<FEATURE_WINDOW_SIZE;fi++)
    {
      for(int fj=0;fj<FEATURE_WINDOW_SIZE;fj++)
      {
        //printf("ii:[%d],jj[%d],fi[%d],fj[%d],scale[%d],INTERVAL[%d],hfsz[%d]\n",ii,jj,fi,fj,scale,INTERVAL,hfsz);
      
        if(ii+fi+1<hfsz || ii+fi+1>row+hfsz || jj+fj+1<hfsz || jj+fj+1>col+hfsz)
	  weight.at<float>(fi,fj) = 0;
	else
	  weight.at<float>(fi,fj) = GT->at<float>(fi,fj)*ipMag[scale/INTERVAL][scale%INTERVAL].at<float>(ii+fi+hfsz,jj+fj+hfsz);

	   //weight.at<float>(fi,fj) = ipMag[scale/INTERVAL][scale%INTERVAL].at<float>(ii+fi+hfsz,jj+fj+hfsz);
      
      }
    }

    for(int i=0;i<4;i++)
    {
      for(int j=0;j<4;j++)
      {
      
        for(int h=0;h<DESC_NUM_BINS;h++)
	  hist[h] = 0.0;
	
	int starti = static_cast<int>(ii) - static_cast<int>(hfsz) + 1 + static_cast<int>((hfsz/2)*i);
	int startj = static_cast<int>(jj) - static_cast<int>(hfsz) + 1 + static_cast<int>((hfsz/2)*j);
	int endi = static_cast<int>(ii) + static_cast<int>((hfsz/2)*(static_cast<int>(i)-1));
	int endj = static_cast<int>(jj) + static_cast<int>((hfsz/2)*(static_cast<int>(j)-1));


	for(int k=starti;k<=endi;k++)
	{
	  for(int l=startj;l<=endj;l++)
	  {
	    if(k<0 || k>row || l<0 || l>col)
	      continue;

	    double sample_orien = ipOrient[scale/INTERVAL][scale%INTERVAL].at<double>(k,l);
	    sample_orien -= main_orien;

	    /*while(sample_orien<0)
	      sample_orien += 2*M_PI;

	    while(sample_orien>2*M_PI)
	      sample_orien -= 2*M_PI;*/

	    if(sample_orien<0)
	      sample_orien += 2*M_PI;

	    if(sample_orien>2*M_PI)
	      sample_orien = M_PI-0.005;

	    if(sample_orien<0)
	      sample_orien = 0.005;


	    if(isnan(sample_orien))
	    {
	      //printf("Sample orientation: <%f>\n",sample_orien);
	      sample_orien = 0;
	    }
	    if(sample_orien<=-0.02 || sample_orien>=2*M_PI+0.02)
	      printf("Sample orien:<%f>",sample_orien);

	    assert(sample_orien>=-0.02 && sample_orien<2*M_PI+0.02);

	    unsigned int sample_orien_d = sample_orien * 180/M_PI;
	    assert(sample_orien<360);

	    unsigned int bin = sample_orien_d/(360/DESC_NUM_BINS);
	    double bin_f = (double)sample_orien_d/(double)(360/DESC_NUM_BINS);

	    assert(bin<DESC_NUM_BINS);
	    assert(k+hfsz-1-ii<FEATURE_WINDOW_SIZE && l+hfsz-1-jj<FEATURE_WINDOW_SIZE);

	    hist[bin]+=(1-fabs(bin_f-(bin+0.5)))*weight.at<float>(k+hfsz-1-ii,l+hfsz-1-jj);

	  }
	}

	for(int v=0;v<DESC_NUM_BINS;v++)
	{
	  fv[(i*FEATURE_WINDOW_SIZE/4+j)*DESC_NUM_BINS+v] = hist[v];
	}
      }
    }

    double norm = 0;
    for(int v=0;v<FV_SIZE;v++)
      norm+=pow(fv[v],2.0);

    norm=sqrt(norm);

    for(int v=0;v<FV_SIZE;v++)
      fv[v]= fv[v] / norm;

    for(int v=0;v<FV_SIZE;v++)
      if(fv[v]>FV_THRESH)
        fv[v] = FV_THRESH;

    norm = 0;
    for(int v=0;v<FV_SIZE;v++)
      norm+=pow(fv[v],2.0);

    norm=sqrt(norm);

    for(int v=0;v<FV_SIZE;v++)
      fv[v]=fv[v]/norm;

    struct KeyDescriptor temp;
    temp.x = descx;
    temp.y = descy;

    temp.fv = fv;

    keyDesc.push_back(temp);
    
  }

}

int Sift::kernelSize(double sigma, double cut_off)
{
  int i=0;
  
  for(i=0;i<MAX_KERNEL_SIZE;i++)
  {
    if (exp(-((double)(i*i))/(2.0*sigma*sigma))<cut_off)
      break;
  }
  
  return (2*i-1);
}

Mat* Sift::buildGuassianTable(int size, double sigma)
{
  double half_kernel_size = size/2-0.5;

  assert(size%2==0);

  Mat *gTable = new Mat(Size(size,size),CV_32F);
  double temp = 0;
  double g_exp = 0;
  for(int i=0;i<size;i++)
  {
    for(int j=0;j<size;j++)
    {
      double x,y;
      x = i-half_kernel_size;
      y = j-half_kernel_size;
      temp = 1.0/(2*M_PI*sigma*sigma) * exp(-(x*x+y*y)/(2.0*sigma*sigma));
      gTable->at<float>(i,j) = temp;
      g_exp += temp;
    }
  }

  for(int i=0;i<size;i++)
  {
    for(int j=0;j<size;j++)
    {
      gTable->at<float>(j,i) = 1/g_exp *gTable->at<float>(j,i);
    }
  }
  return gTable;

}

void Sift::drawKeyPoints()
{
  Mat disp = input.clone();

  for(int i=0;i<n_keyPoints;i++)
  {
    struct KeyPoint kp = keyPoints[i];
    circle(disp,Point(kp.y,kp.x),2,Scalar(0,0,255));
    //line(disp,Point(kp.x,kp.y),Point(kp.x+10*cos(kp.ori[0]),kp.y+10*sin(kp.ori[0])),Scalar(0,255,0),2,8);

  }

  namedWindow("SIFT Keypoints",1);
  imshow("SIFT Keypoints",disp);
}

int main( int argc, char ** argv)
{
  CommandLineParser parser(argc,argv,"{help h | |}{@image|..images/something.jpg|}");
  if (parser.has("help"))
  {
    std::cout << "Give the image name along with the command\n";
    return 0;
  }

  //std::string imagename = parser.get<std::string>("@image");
  char *imagename = argv[1];
  Sift s(imagename);
  s.drawKeyPoints();

  waitKey(0);

  return 0;
}
