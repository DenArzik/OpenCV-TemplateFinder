#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;


const char *src_path("../../TemplateFinder/resources/dogs.jpg");
const char *tmpl_path("../../TemplateFinder/resources/template.jpg");

const char *src_name("Source");
const char *tmpl_name("Template");
const char *result_name("Result");

const int max_Trackbar(5);
int match_method(0);

Mat src_img;
Mat tmpl_img;

// Function Headers
void MatchingMethod(int, void*);


/** @function main */
int main()
{
	namedWindow(src_name, WINDOW_NORMAL);
	namedWindow(tmpl_name, WINDOW_NORMAL);
	namedWindow(result_name, WINDOW_NORMAL);

	resizeWindow(src_name, 600, 300);
	resizeWindow(result_name, 600, 300);

	src_img = imread(src_path);
	tmpl_img = imread(tmpl_path);

	imshow(tmpl_name, tmpl_img);

	// Create Trackbar
	const char *trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
	createTrackbar(trackbar_label, src_name, &match_method, max_Trackbar, MatchingMethod);

	MatchingMethod(0, 0);

	waitKey(0);
	return 0;
}

/**
* @function MatchingMethod
* @brief Trackbar callback
*/
void MatchingMethod(int, void*)
{
	// Source image to display
	Mat result_img;
	src_img.copyTo(result_img);

	// Create the result matrix
	int result_cols = src_img.cols - tmpl_img.cols + 1;
	int result_rows = src_img.rows - tmpl_img.rows + 1;

	// Do the Matching and Normalize
	matchTemplate(src_img, tmpl_img, result_img, match_method);
	normalize(result_img, result_img, 0, 1, NORM_MINMAX, -1, Mat());

	// Localizing the best match with minMaxLoc
	Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result_img, 0,0, &minLoc, &maxLoc, Mat());

	// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	src_img = imread(src_path);

	// Show me what you got
	rectangle(src_img, matchLoc, Point(matchLoc.x + tmpl_img.cols, matchLoc.y + tmpl_img.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result_img, matchLoc, Point(matchLoc.x + tmpl_img.cols, matchLoc.y + tmpl_img.rows), Scalar::all(0), 2, 8, 0);
	
	imshow(src_name, src_img);
	imshow(result_name, result_img);

	return;
}
