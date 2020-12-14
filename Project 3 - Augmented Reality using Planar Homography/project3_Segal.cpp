/*
Matan Segal
CS 5390
Projet 3 - Augmented Reality using Planar Homography
Report can be found in the file project3_writeUp_Segal.pdf
*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main(){

    Mat img_1 = imread("Project3/box_in_scene.png"); // need to be the desk image!
    Mat img_2 = imread("Project3/box.png"); // need to be the object image!

    if (img_1.empty() || img_2.empty()){
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    // Part 1 - Matching Points
    // took and modified from: https://github.com/sunzuolei/orb/blob/master/feature_extration.cpp

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // corner detectors 
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // BRIEF calculation based on the location of the corner
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    // Hamming distance matching
    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);

    double min_dist = 10000, max_dist = 0;

    // Find the minimum and maximum distances between all matches, 
    // the distance between the most similar and least similar two sets of points
    for (int i = 0; i < descriptors_1.rows; i++){
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // setting a threshold for saving just the good matches
    std::vector< DMatch > good_matches;
    float ratio = 2.5;

    double thresh = max(ratio * min_dist, 30.0);

    for (int i = 0; i < descriptors_1.rows; i++){
        if (matches[i].distance < thresh)
            good_matches.push_back(matches[i]);
        
    }
    
    // show the good matches we found 
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    imshow("img_goodmatch", img_goodmatch);

    // ------------------------------------------------------
    // Part 2,3 - Warp with a Homography
    // took and changed from: https://www.learnopencv.com/homography-examples-using-opencv-python-c/
    
    int num_matches = good_matches.size();
    vector<Point2f> matched_points1;
    vector<Point2f> matched_points2;


    // getting the correspoinding points
    // https://stackoverflow.com/questions/12937490/how-to-access-points-location-on-opencv-matcher
    for (int i = 0;i < num_matches;i++){
        int idx2 = good_matches[i].trainIdx;
        int idx1 = good_matches[i].queryIdx;
        matched_points1.push_back(keypoints_1[idx1].pt);
        matched_points2.push_back(keypoints_2[idx2].pt);
    }

    // calculate Homograpth
    Mat H = findHomography(matched_points2, matched_points1, RANSAC);
    Mat homograpth_out;

    // Warp source image to destination based on homography
    warpPerspective(img_2, homograpth_out, H, img_1.size()); 

    imshow("Warped Source Image", homograpth_out);

    // convert the frame of the book to the desk image
    // https://docs.opencv.org/master/d9/dab/tutorial_homography.html

    std::vector<Point> corners_source = { Point(0,0), Point(0,img_2.rows - 1),
                                        Point(img_2.cols - 1, img_2.rows - 1),Point(img_2.cols - 1,0) };

    // multiply the point by the H matrix
    std::vector<Point> corners_dst;
    for (auto c : corners_source) {
        cout <<"source: " << c << endl;
        Mat pt1 = (Mat_<double>(3, 1) << c.x, c.y, 1);
        Mat pt2 = H * pt1;        // X' = H X
        pt2 /= pt2.at<double>(2); // so it will be 1 on the last row
        corners_dst.push_back(Point((int)(pt2.at<double>(0)), (int)pt2.at<double>(1)));
    }

    // calculate bounderies of the frame, for smaller range
    int highest = min(corners_dst[0].y, corners_dst[3].y);
    int lowest = max(corners_dst[2].y, corners_dst[1].y);
    int leftest = min(corners_dst[0].x, corners_dst[1].x);
    int rightest = max(corners_dst[3].x, corners_dst[2].x);


    //-------------------------------------------------------
    // Part 4 - Bring It to Life with video
    // read and display: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

    VideoCapture video("Project3/ar_source.mov");

    // Check if video opened successfully
    if (!video.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;  
    }

    while (1) {
        Mat frame;
        // Capture frame-by-frame
        video >> frame;

        // If the frame is empty, break 
        if (frame.empty())
            break;

        // wrap frame to the desk perspective
        resize(frame, frame, img_2.size());
        Mat homography_frame;
        warpPerspective(frame, homography_frame, H, img_1.size());

        //printing homography (the video with perspective) on the desk img (img_1)
        for (int i = highest; i < lowest; ++i) {
            for (int j = leftest; j < rightest; ++j) {
                if (homograpth_out.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
                    img_1.at<Vec3b>(i, j) = homography_frame.at<Vec3b>(i, j);
            }
        }

        // Display the resulting frame
        imshow("Frame", img_1);

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(25);
        if (c == 27)
            break;
    }

    video.release();
        
    waitKey(0);

    return 0;
}