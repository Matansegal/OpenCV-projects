/*
Matan Segal
CS 5390
Projet 3 - Bonus
based on the previous code - project3_Segal
*/


#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

using namespace std;
using namespace cv;

Mat homograpth_out, H;
Mat homography_frame, frame;
Mat img_1, img_2, out;

thread t1;
double thresh = 150;
int counter;

std::vector<KeyPoint> keypoints_1, keypoints_2;
Mat descriptors_1, descriptors_2;
Ptr<FeatureDetector> detector = ORB::create();
Ptr<DescriptorExtractor> descriptor = ORB::create();
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

// for the forEach funciton, to iterate faster
// from: https://answers.opencv.org/question/197369/matforeach-operation-is-resulting-in-error/
struct Operator {
    void operator ()(Vec3b& pixel, const int* position) const
    {
        if (homograpth_out.at<Vec3b>(position[0], position[1]) != Vec3b(0, 0, 0))
            pixel = homography_frame.at<Vec3b>(position[0], position[1]);
        else
            pixel = img_1.at<Vec3b>(position[0], position[1]);
    }
};


void compute_perspective() {
    // corner detectors 
    detector->detect(img_1, keypoints_1);
    // BRIEF calculation based on the location of the corner
    descriptor->compute(img_1, keypoints_1, descriptors_1);

    // just the first time, since this is the object image, which doesn't change
    if (counter == 0) {
        detector->detect(img_2, keypoints_2);
        descriptor->compute(img_2, keypoints_2, descriptors_2);
    }

    // Hamming distance matching
    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);

    // setting a threshold for saving just the good matches
    vector<Point2f> matched_points1;
    vector<Point2f> matched_points2;

    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance < thresh) {
            matched_points1.push_back(keypoints_1[matches[i].queryIdx].pt);
            matched_points2.push_back(keypoints_2[matches[i].trainIdx].pt);
        }
    }

    // calculate Homograpth
    H = findHomography(matched_points2, matched_points1, RANSAC);

    // Warp source image to destination based on homography
    warpPerspective(img_2, homograpth_out, H, img_1.size());

    //cout << "thread done -- " << counter << endl;
}


int main(){

    img_2 = imread("Project3/cv_cover.jpg"); // need to be the object image!

    if (img_2.empty()){
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    VideoCapture cover_video("Project3/book.mov");
    VideoCapture video("Project3/ar_source.mov");

    // Check if video opened successfully
    if (!cover_video.isOpened() || !video.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    counter = 0;
    
    while (1) {
        
        // frame of the cover video is the same as img_1 from previuos code
        cover_video >> img_1;
        out = img_1.clone();

        if (img_1.empty())
            break;

        // first frame
        if (counter == 0)
            compute_perspective();

        // check for new perspective just once in 7 frames
        // use other thread for calculate perspective
        else if (counter % 7 == 0)
            t1 = thread(compute_perspective);

        // join it after 7 frames
        else if (counter % 7 == 6 && counter != 6)
            t1.join();

        // Capture frame of the video to display
        video >> frame;

        // If the frame is empty, break 
        if (frame.empty())
            break;

        // wrap frame to the desk perspective
        resize(frame, frame, img_2.size());
        
        warpPerspective(frame, homography_frame, H, img_1.size());

        //printing homography (the video with perspective) on the desk img (img_1)
        out.forEach<Vec3b>(Operator());

        // Display the resulting frame
        imshow("Frame", out);

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(25);
        if (c == 27)
            break;

        counter++;
    }

    cout << "done.." << endl;

    video.release();
        
    waitKey(0);

    return 0;
}