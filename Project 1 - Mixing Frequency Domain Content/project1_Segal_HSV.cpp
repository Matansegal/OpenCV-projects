/*
Matan Segal
Project 1
Fake Images - Mixing Frequency Domain Content
*/


#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

const double PI = acos(-1);
// all the 1's are for the low pass
// all the 2's for the high pass
Mat img1;
Mat img2;
Mat channels1[3];
Mat channels2[3];
Mat FTD_image1;
Mat FTD_image2;
Mat filterd1;
Mat filterd2;
Mat output1;
Mat output2;
Mat kernal1; // low pass
Mat kernal2; // high pass
Mat combined;
int sigma1 = 10; // for low pass
int sigma2 = 10; // for high pass
int percentagePic1 = 50;
int opt = 0; // 2 options to switch between img1 and img2
string pic1; // name of files
string pic2;


bool processImage(string pic, Mat& img, Mat channels[], Mat& kernal, Mat& FTD_image, Mat& filterd, Mat& output, int& sigma, bool highpass);
double gaussian(double x, double y, int& sigma);
void produce2dGaussianKernel(Mat& kernal, int width, int hight, int& sigma, bool highpass = false);
void changeQuadrants(Mat& kernal);
void changeToOptimalDFTsize(Mat& img, Mat& dst);
void calcDFT(Mat& src, Mat& dst);
void filtering(Mat& src, Mat& dst, Mat& filter);
void on_trackbar_sigma1(int, void*);
void on_trackbar_sigma2(int, void*);
void on_trackber_ratio(int, void*);
void on_trackber_opt(int, void*);

void print(Mat& m);


int main(int argc, const char** argv){

    pic1 = "Project1_Data/cat.jpg";
    pic2 = "Project1_Data/dog.jpg";

    if (!processImage(pic1, img1, channels1, kernal1, FTD_image1, filterd1, output1, sigma1, false))
        return -1;

    if (!processImage(pic2, img2, channels2, kernal2, FTD_image2, filterd2, output2, sigma2, true))
        return -1;

    resize(output1, output1, Size(512, 512));
    resize(output2, output2, Size(512,512));

    // should try to normalize 0,255
    // also try to do invers Gaussain as 0-val
    combined = 0.5 * output1 + 0.5 * output2;

    namedWindow("combained pics");
    imshow("combained pics", combined);

    createTrackbar("sigma1", "combained pics", &sigma1, 100, on_trackbar_sigma1);
    createTrackbar("sigma2", "combained pics", &sigma2, 100, on_trackbar_sigma2);
    createTrackbar("percentagePic1", "combained pics", &percentagePic1, 100, on_trackber_ratio);
    createTrackbar("option", "combained pics", &opt,1, on_trackber_opt);


    int count = 1;
    for (;;) {
        char key = (char)waitKey(30);
        if (key == 's') {
            imwrite("result_p1_Segal" + to_string(count) + ".png", combined);
            count++;
        }

        else if (key == 27) // if esc
            break;
    }


    return 0;

}

// main process calling all the fucntion for the two pics
bool processImage(string pic, Mat& img, Mat channels[], Mat& kernal, Mat& FTD_image, Mat& filterd, Mat& output, int& sigma, bool highpass) {
    
    img = imread(pic); //, IMREAD_GRAYSCALE
    if (img.empty()){
        printf("Cannot read %s\n", pic);
        return false;
    }
    // in order to name different the windows
    string num = highpass ? "2" : "1";

    // show src image
    namedWindow("src"+num);
    imshow("src"+num, img);

    changeToOptimalDFTsize(img, img);
    
    
    // convert to HSV
    Mat hsv;
    cvtColor(img, hsv, COLOR_RGB2HSV);
    split(hsv, channels);

    // Fourie transform of channel 2 since this is the sominant channel
    channels[2].convertTo(channels[2], CV_32FC1, 1.0);
    calcDFT(channels[2], FTD_image);

    // creating the gaussian filter
    produce2dGaussianKernel(kernal, img.rows, img.cols, sigma1, highpass);

    // show before changing the quadrants
    namedWindow("filter"+num);
    imshow("filter"+num, kernal);

    filtering(FTD_image, filterd, kernal);

    // inverse dft
    dft(filterd, output, DFT_INVERSE | DFT_REAL_OUTPUT);
    normalize(output, output, 0, 1, NORM_MINMAX);
    output.convertTo(output, CV_8UC1, 255.0);
    channels[2] = output;

    // merging back to hsv form
    merge(channels, 3, hsv);
    cvtColor(hsv, output, COLOR_HSV2RGB);

    return true;
}

// calculate the gaussion value
double gaussian(double x, double y, int& sigma) {
    double sig_sqr = sigma * sigma;
    return (1. / (2 * PI * sig_sqr)) * exp(-0.5 * (x * x + y * y) / sig_sqr);
}

// modified from:
// https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel
// creating a gaussian filter in the size of the image in order to multiply the image's dft with the kernal
void produce2dGaussianKernel(Mat& kernal, int width, int hight, int& sigma, bool highpass) {
    double Xcenter = width % 2 == 1 ? width / 2 : width / 2 - 0.5;
    double Ycenter = hight % 2 == 1 ? hight / 2 : hight / 2 - 0.5;
    kernal = Mat::zeros(width, hight, CV_32F);
    // compute values
    for (int i = 0; i < width; i++)
        for (int j = 0; j < hight; j++) {
            double val = gaussian(i - Xcenter, j - Ycenter, sigma); // x = 1 - center , y = j - center
            kernal.at<float>(i, j) = val;
        }
    // normalize
    normalize(kernal, kernal, 0, 1, NORM_MINMAX);
    // for highpass invert it 
    if (highpass)
        kernal = 1 - kernal;

}


// since the dft is in its original form, I need to change the qudrant of the kernel
void changeQuadrants(Mat& kernal) {

    int cx = kernal.cols / 2;
    int cy = kernal.rows / 2;

    // rearrange the quadrants of filter, so it will be like the dft Mat
    Mat tmp;
    Mat q0(kernal, Rect(0, 0, cx, cy));
    Mat q1(kernal, Rect(cx, 0, cx, cy));
    Mat q2(kernal, Rect(0, cy, cx, cy));
    Mat q3(kernal, Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}



void changeToOptimalDFTsize(Mat& img, Mat& dst) {
    int M = getOptimalDFTSize(img.rows);
    int N = getOptimalDFTSize(img.cols);
    Mat padded;
    copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0)); // padded is the new img with the borders
    dst = padded;
}

void calcDFT(Mat& src, Mat& dst) {
    Mat planes[] = { Mat_<float>(src), Mat::zeros(src.size(), CV_32F) };
    Mat complexImg;
    merge(planes, 2, complexImg);

    dft(complexImg, complexImg); // (src,dst)
    dst = complexImg;
}

void filtering(Mat& src, Mat& dst, Mat& filter) {
    Mat planes_dft[] = { src, Mat::zeros(src.size(), CV_32F) };
    split(src, planes_dft);

    // change the quadrants of the filter
    changeQuadrants(filter);

    Mat planes_out[] = { Mat::zeros(src.size(), CV_32F),Mat::zeros(src.size(), CV_32F) };
    planes_out[0] = filter.mul(planes_dft[0]);
    planes_out[1] = filter.mul(planes_dft[1]);

    merge(planes_out, 2, dst);

}

// changing the sigma of the low pass filter
void on_trackbar_sigma1(int, void*){
    produce2dGaussianKernel(kernal1, img1.rows, img1.cols, sigma1, false);

    imshow("filter1", kernal1);

    filtering(FTD_image1, filterd1, kernal1);

    dft(filterd1, output1, DFT_INVERSE | DFT_REAL_OUTPUT);
    normalize(output1, output1, 0, 1, NORM_MINMAX);
    output1.convertTo(output1, CV_8UC1, 255.0);
    channels1[2] = output1;

    // merging back to hsv form
    Mat hsv;
    merge(channels1, 3, hsv);
    cvtColor(hsv, output1, COLOR_HSV2RGB);

    resize(output1, output1, Size(512, 512));

    double p1 = double(percentagePic1) / 100;
    double p2 = 1 - p1;
    combined = p1 * output1 + p2 * output2;

    imshow("combained pics", combined);
}

// chagning the sigma of the high pass filter
void on_trackbar_sigma2(int, void*) {
    produce2dGaussianKernel(kernal2, img2.rows, img2.cols, sigma2, true);

    imshow("filter2", kernal2);

    filtering(FTD_image2, filterd2, kernal2);

    dft(filterd2, output2, DFT_INVERSE | DFT_REAL_OUTPUT);
    normalize(output2, output2, 0, 1, NORM_MINMAX);
    output2.convertTo(output2, CV_8UC1, 255.0);
    channels2[2] = output2;

    // merging back to hsv form
    Mat hsv;
    merge(channels2, 3, hsv);
    cvtColor(hsv, output2, COLOR_HSV2RGB);

    resize(output2, output2, Size(512, 512));

    double p1 = double(percentagePic1) / 100;
    double p2 = 1 - p1;
    combined = p1 * output1 + p2 * output2;

    imshow("combained pics", combined);
}

void on_trackber_ratio(int, void*){
    double p1 = double(percentagePic1) / 100;
    double p2 = 1 - p1;
    combined = p1 * output1 + p2 * output2;

    imshow("combained pics", combined);
}

// switching between pic1 and pic2
void on_trackber_opt(int, void*) {
    string tmp = pic1;
    pic1 = pic2;
    pic2 = tmp;

    processImage(pic1, img1, channels1, kernal1, FTD_image1, filterd1, output1, sigma1, false);
    processImage(pic2, img2, channels2, kernal2, FTD_image2, filterd2, output2, sigma2, true);

    resize(output1, output1, Size(512, 512));
    resize(output2, output2, Size(512, 512));

    double p1 = double(percentagePic1) / 100;
    double p2 = 1 - p1;
    combined = p1 * output1 + p2 * output2;

    namedWindow("combained pics");
    imshow("combained pics", combined);
}




// for testing
void print(Mat& m) {
    double s = 0;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            printf("%.5f ", m.at<float>(i, j));
            s += m.at<float>(i, j);
        }
        printf("\n");
    }
    printf("sum:%f\n", s);
}