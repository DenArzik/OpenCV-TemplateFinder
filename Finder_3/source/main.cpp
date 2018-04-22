#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/legacy/legacy.hpp"
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	//read images
	Mat img_1c = imread("../../Finder_3/resources/5.jpg");
	Mat img_2c = imread("../../Finder_3/resources/6.jpg");
	Mat img_1, img_2;
	//transform images into gray scale
	cvtColor(img_1c, img_1, CV_BGR2GRAY);
	cvtColor(img_2c, img_2, CV_BGR2GRAY);
	
	SIFT sift(50, 5);
	vector<KeyPoint> key_points_1, key_points_2;
	Mat detector;
	//do sift, find key points
	sift(img_1, Mat(), key_points_1, detector);
	sift(img_2, Mat(), key_points_2, detector);
	
	SiftDescriptorExtractor extractor;
	Mat descriptors_1, descriptors_2;
	//compute descriptors
	extractor.compute(img_1, key_points_1, descriptors_1);
	extractor.compute(img_2, key_points_2, descriptors_2);
	
	//use burte force method to match vectors
	BruteForceMatcher<L2<float> >matcher;
	vector<DMatch>matches;
	matcher.match(descriptors_1, descriptors_2, matches);
	
	//draw results
	Mat img_matches;
	drawMatches(img_1c, key_points_1, img_2c, key_points_2, matches, img_matches);
	imshow("sift_Matches", img_matches);
	waitKey(0);
	return 0;
}
