#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <limits>
#include "kmboxNet.h"
#include "pid.h"

static int lostFrames = 0;

inline void FindColor(const cv::Mat& src) {
    std::vector<cv::Point2f> centers;
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask1, mask2, mask;
    cv::Scalar lower1(0, 100, 120), upper1(3, 255, 255);
    cv::Scalar lower2(177, 100, 100), upper2(180, 255, 255);
    cv::inRange(hsv, lower1, upper1, mask1);
    cv::inRange(hsv, lower2, upper2, mask2);
    mask = mask1 | mask2;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    const double MIN_AREA = 2.0;
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < MIN_AREA) continue;
        cv::Moments m = cv::moments(c);
        if (m.m00 == 0) continue;
        cv::Point2f center(
            static_cast<float>(m.m10 / m.m00),
            static_cast<float>(m.m01 / m.m00)
        );
        //if (center.x < 60 || center.x > 100)continue;
        centers.push_back(center);
    }

    // ---- 找到离 (80,80) 最近的点 ----
    cv::Point2f target(80.0f, 80.0f);
    cv::Point2f nearest(-1.0f, -1.0f); // -1,-1 表示没找到
    float minDistSq = std::numeric_limits<float>::max();

    for (const auto& pt : centers) {
        float dx = pt.x - target.x;
        float dy = pt.y - target.y;
        float distSq = dx * dx + dy * dy;
        if (distSq < minDistSq) {
            minDistSq = distSq;
            nearest = pt;
        }
    }

	//std::cout << nearest << std::endl;

    if (nearest.x < 0 && nearest.y < 0) {
        lostFrames++;
        if (lostFrames >= 5) {
            CenterToScreen(1280.0f, 800.0f);
        }
        return; // 没找到有效点
    }
    lostFrames = 0;

    int x = static_cast<int>(nearest.x) + 1200;
    int y = static_cast<int>(nearest.y) + 720;
    if (kmNet_monitor_mouse_right()) {
		if (abs(1280 - x) < 3) x = 1280;
		if (abs(800 - y) < 3) y = 800;
		x -= 1280;
        x *= 0.7;
        x += 1280;
        y -= 800;
		y *= 0.7;
		y += 800;
        std::cout << x << " " << y << std::endl;
        CenterToScreen(static_cast<float>(x), static_cast<float>(y));
    }
	else CenterToScreen(1280.0f, 800.0f);
}