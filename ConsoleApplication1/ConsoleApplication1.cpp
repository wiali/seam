// ConsoleApplication1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    cv::namedWindow("result");

    Mat whole_image = imread("aa.jpg");
    Mat img = whole_image.clone();

    cv::Mat edge(img.rows, img.cols, CV_8UC1);
    cv::Mat gray(img.rows, img.cols, CV_8UC1);
    cv::cvtColor(img, gray, CV_BGR2GRAY); //Convert to gray

    cv::threshold(gray, edge, 0, 255, cv::THRESH_BINARY); //Threshold the gray
    
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(edge, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    
    for (auto iter = contours.begin(); iter != contours.end(); )
    {
        double area = contourArea(*iter, false);  //  Find the area of contour
        if (fabs(area) < 100)
            iter = contours.erase(iter);
        else
            ++iter;
    }

    cv::Mat contour_img(img.rows, img.cols, CV_8UC1);
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(255, 255, 255);
        drawContours(contour_img, contours, (int)i, color, 5, 8, hierarchy, 0, cv::Point());
    }

    imshow("contours", contour_img);
    cv::waitKey(0);

    img = contour_img;

    whole_image.convertTo(whole_image, CV_32FC3, 1.0 / 255.0);
    cv::resize(whole_image, whole_image, img.size());
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    Mat bg = Mat(img.size(), CV_32FC3);
    bg = Scalar(1.0, 1.0, 1.0);

    // Prepare mask
    Mat mask;
    Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    img_gray.convertTo(mask, CV_32FC1);
    threshold(1.0 - mask, mask, 0.9, 1.0, cv::THRESH_BINARY_INV);

    cv::GaussianBlur(mask, mask, Size(21, 21), 11.0);
    imshow("result", mask);
    cv::waitKey(0);


    // Reget the image fragment with smoothed mask
    Mat res;

    vector<Mat> ch_img(3);
    vector<Mat> ch_bg(3);
    cv::split(whole_image, ch_img);
    cv::split(bg, ch_bg);
    ch_img[0] = ch_img[0].mul(mask) + ch_bg[0].mul(1.0 - mask);
    ch_img[1] = ch_img[1].mul(mask) + ch_bg[1].mul(1.0 - mask);
    ch_img[2] = ch_img[2].mul(mask) + ch_bg[2].mul(1.0 - mask);
    cv::merge(ch_img, res);
    cv::merge(ch_bg, bg);

    imshow("result", res);
    cv::waitKey(0);
    cv::destroyAllWindows();
}