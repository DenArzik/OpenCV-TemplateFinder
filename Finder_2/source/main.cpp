#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

const static float minHessian = 400.f;
const static float magicCoef = 3.f;

void detect(const cv::Mat &im1, std::vector<cv::KeyPoint> &kp1, const cv::Mat &im2, std::vector<cv::KeyPoint> &kp2)
{
	cv::SurfFeatureDetector detector(minHessian);
	detector.detect(im1, kp1);
	detector.detect(im2, kp2);
}
void extract(const cv::Mat &im1, std::vector<cv::KeyPoint> &kp1, cv::Mat &dsc1, const cv::Mat &im2, std::vector<cv::KeyPoint> &kp2, cv::Mat &dsc2)
{
	cv::SurfDescriptorExtractor extractor;
	extractor.compute(im1, kp1, dsc1);
	extractor.compute(im2, kp2, dsc2);
}
void match(const cv::Mat &dsc1, const cv::Mat &dsc2, std::vector<cv::DMatch> &matches)
{
	cv::FlannBasedMatcher matcher;
	matcher.match(dsc1, dsc2, matches);
}

int main()
{
	//cv::VideoCapture cam(0);
	//if (!cam.isOpened())
	//{
	//	std::cout << "NE RABOTAET KAMERA!\n";
	//}
	cv::namedWindow("window", cv::WINDOW_AUTOSIZE);

	cv::Mat img_obj = cv::imread("../../Finder_2/resources/template.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat img_scene = cv::imread("../../Finder_2/resources/dogs.jpg", CV_LOAD_IMAGE_GRAYSCALE);

		
	while (true)
	{
		//cam >> img_scene;
		//cv::resize(img_scene, img_scene, cv::Size(620, 480));
		//cv::flip(img_scene, img_scene, 1);

		std::vector<cv::KeyPoint>  kp_obj, kp_scene;
		detect(img_obj, kp_obj, img_scene, kp_scene);

		cv::Mat dsc_obj, dsc_scene;
		extract(img_obj, kp_obj, dsc_obj, img_scene, kp_scene, dsc_scene);
			
		if (dsc_scene.empty()) continue;
		
		std::vector<cv::DMatch> matches;
		match(dsc_obj, dsc_scene, matches);

		float maxDist = 0.f;
		float minDist = 100.f;

		for (int i = 0; i < dsc_obj.rows; ++i)
		{
			double dist = matches[i].distance;
			if (dist < minDist) minDist = dist;
			else if (dist > maxDist) maxDist = dist;
		}

		std::vector<cv::DMatch> goodMatches;

		for (int i = 0; i < dsc_obj.rows; ++i)
			if (matches[i].distance < magicCoef*minDist)
				goodMatches.push_back(matches[i]);

		cv::Mat imgMatches;
		cv::drawMatches(img_obj, kp_obj, img_scene, kp_scene, goodMatches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		
		std::vector<cv::Point2f> obj;
		std::vector<cv::Point2f> scene;

		for (int i(0); i < goodMatches.size(); ++i)
		{
			obj.push_back(kp_obj[goodMatches[i].queryIdx].pt);
			scene.push_back(kp_scene[goodMatches[i].trainIdx].pt);
		}
		cv::Mat H;
		try
		{
			H = cv::findHomography(obj, scene, cv::FM_RANSAC);
		}
		catch (...)
		{
			continue;
		}
	
		std::vector<cv::Point2f> obj_corners(4), scene_corners(4);

		obj_corners[0] = cv::Point(0, 0);
		obj_corners[0] = cv::Point(img_obj.cols, 0);
		obj_corners[0] = cv::Point(img_obj.cols, img_obj.rows);
		obj_corners[0] = cv::Point(0, img_obj.rows);

		cv::perspectiveTransform(obj_corners, scene_corners, H);

		cv::line(imgMatches, scene_corners[0] + cv::Point2f(img_obj.cols, 0), scene_corners[1] + cv::Point2f(img_obj.cols, 0), cv::Scalar(0, 255, 0), 4);
		cv::line(imgMatches, scene_corners[1] + cv::Point2f(img_obj.cols, 0), scene_corners[2] + cv::Point2f(img_obj.cols, 0), cv::Scalar(0, 255, 0), 4);
		cv::line(imgMatches, scene_corners[2] + cv::Point2f(img_obj.cols, 0), scene_corners[3] + cv::Point2f(img_obj.cols, 0), cv::Scalar(0, 255, 0), 4);
		cv::line(imgMatches, scene_corners[3] + cv::Point2f(img_obj.cols, 0), scene_corners[0] + cv::Point2f(img_obj.cols, 0), cv::Scalar(0, 255, 0), 4);


		cv::imshow("window", imgMatches);
		cv::waitKey(30);
	}

	
	std::cin.get();
	return 0;
}