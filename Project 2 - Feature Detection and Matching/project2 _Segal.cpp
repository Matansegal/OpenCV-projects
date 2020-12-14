#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;


const int sum_thresh = 800;
vector<Point> corners1 , corners2;
Mat img1, img2;
vector<Mat> kernels1, kernels2;
vector<pair<pair<Point,Point>, int>> corralate_corners; // holds <<corner1,corner2>,diff value>


void find_corners( Mat& img, const int& thresh, vector<Point>& vec);
void set_mat_vector(Mat& src, vector<Mat>& kernels, vector<Point>& corners);
void set_corralate_corners();
void draw_circles();
void draw_connection_lines(Mat& out, int div);
void resizing(Mat& img, Mat& longer_img);


int main( int argc, char** argv ){

    img1 = imread("Project3/hp_desk.png");
    img2 = imread("Project3/hp_cover.jpg");

    if ( img1.empty() || img2.empty()) {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    // part 1
    const int thresh1 = 90;
    const int thresh2 = 110;
    find_corners(img1,  thresh1, corners1);
    find_corners(img2,  thresh2, corners2);

    // part 2
    set_mat_vector(img1, kernels1, corners1);
    set_mat_vector(img2, kernels2, corners2);

    // part 3
    set_corralate_corners();

    // demonstartion
    draw_circles();
    
    Mat out, out1;
    int div_len = 10; // the border between the iamges
    Mat divider = Mat(Size(div_len,img1.rows), img1.type(),Scalar(255,255,255));

    hconcat(img1, divider, out1);

    // resizing img2 if it is shorter than img1, and the oposite
    if (img2.rows < img1.rows)
        resizing(img2, img1);
    else if(img2.rows > img1.rows)
        resizing(img1, img2);

    hconcat(out1, img2, out);

    draw_connection_lines(out, div_len);

    namedWindow("output");
    imshow("output", out);
    //imwrite("project2/output.jpg", out);
    

    waitKey();
    return 0;
}



void find_corners( Mat& img, const int& thresh, vector<Point>& vec){

    // this vector is a parallel vecetor of gradient values for each point
    vector<int> gradient_vals;
    Mat src_gray;
    cvtColor(img, src_gray, COLOR_BGR2GRAY);

    int blockSize = 2;    // H = 2x2 mat
    int apertureSize = 3; // Sobel = 3x3 mat
    double k = 0.04;
    Mat dst = Mat::zeros( img.size(), CV_32FC1 );

    cornerHarris( src_gray, dst, blockSize, apertureSize, k );

    normalize(dst, dst, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    //convertScaleAbs( dst_norm, out);

    // for loops to draw circle aroud point larger than threshold
    for( int i = 0; i < dst.rows ; i++ ){
        for( int j = 0; j < dst.cols; j++ ){
            
            int grad_val = (int)dst.at<float>(i, j);
            if (grad_val > thresh) {
                // check local maximuma in the 7x7 frame
                bool flag = false;
                Point p = Point(j, i);

                for (int k = 0; k < vec.size(); ++k) {
                    if (abs(vec.at(k).x - p.x) < 4 && abs(vec.at(k).y - p.y) < 4) {
                        flag = true;
                        if (grad_val > gradient_vals.at(k)) {
                            vec.at(k) = p;
                            gradient_vals.at(k) = grad_val;
                        }
                        break;
                    }
                }
                if (!flag) {
                    vec.push_back(p);
                    gradient_vals.push_back(grad_val);
                }
            }
        }
    }
}


void set_mat_vector(Mat& src, vector<Mat>& kernels, vector<Point>& corners) {
    for (auto p : corners) {
        int size = 5;
        Mat tmp = Mat(size, size, src.type());
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                int y = p.y + i - size / 2;
                int x = p.x + j - size / 2;
                // check bonderies
                if (x < 0 || y < 0 || y >= src.rows || x >= src.cols) {
                    continue;
                }
                else {
                    tmp.at<Vec3b>(i, j)[0] = src.at<Vec3b>(y, x)[0];
                    tmp.at<Vec3b>(i, j)[1] = src.at<Vec3b>(y, x)[1];
                    tmp.at<Vec3b>(i, j)[2] = src.at<Vec3b>(y, x)[2];
                }  
            }
        }
        kernels.push_back(tmp);
    }
}

void set_corralate_corners() {

    for (int i = 0; i < kernels1.size(); ++i) {

        int min_diff = 9999999;
        Point min_diff_p;
        Mat diff;
        for (int j = 0; j < kernels2.size(); ++j) {
            // abs difference
            absdiff(kernels1[i], kernels2[j], diff);

            // sum of the diffs
            int s = cv::sum(cv::sum(diff))[0];

            if (s < min_diff) {
                min_diff = s;
                min_diff_p = corners2[j];
            }

        }
        if (min_diff < sum_thresh) {
            bool flag = true;
            for (int i = 0; i < corralate_corners.size(); ++i) {
                // if the same second point and the diff is smaller, replace with the new one
                if (min_diff_p == corralate_corners.at(i).first.second) {
                    flag = false;
                    if(min_diff < corralate_corners.at(i).second)
                        corralate_corners.at(i) = make_pair(make_pair(corners1[i], min_diff_p), min_diff);
                    break;
                }
            }
            if (flag) {
                corralate_corners.push_back(make_pair(make_pair(corners1[i], min_diff_p ), min_diff));
            }
               
        }
    }

    
    for (auto p : corralate_corners) {
        cout << "p1: " << p.first.first << " , " << "p2: " << p.first.second <<" || " << "diff: " << p.second << endl;
    }
}


void draw_circles() {
    for (auto p : corners1) 
        circle(img1, p, 5, Scalar(0, 0, 255), 2, 8, 0);

    for (auto p : corners2)
        circle(img2, p, 5, Scalar(0, 0, 255), 2, 8, 0);
}

void draw_connection_lines(Mat& out, int div) {
    for (int i = 0; i < corralate_corners.size(); ++i) {
        // adding the horizontal to the second image
        Point sec_p = Point(corralate_corners[i].first.second.x + div + img1.cols, corralate_corners[i].first.second.y);
        line(
            out,                         // Draw onto this image
            corralate_corners[i].first.first,    // Starting here
            sec_p,                       // Ending here
            cv::Scalar(0, 255, 0),       // This color
            1,                           // This many pixels wide
            cv::LINE_AA                  // Draw line in this style
        );
    }
}

void resizing(Mat& img, Mat& longer_img) {
    Mat tmp = img.clone();
    img = Mat::zeros(longer_img.rows, tmp.cols, tmp.type());
    for (int i = 0; i < tmp.rows; ++i) {
        for (int j = 0; j < tmp.cols; ++j) {
            img.at<Vec3b>(i, j)[0] = tmp.at<Vec3b>(i, j)[0];
            img.at<Vec3b>(i, j)[1] = tmp.at<Vec3b>(i, j)[1];
            img.at<Vec3b>(i, j)[2] = tmp.at<Vec3b>(i, j)[2];
        }
    }
}