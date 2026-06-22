#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <limits>
#include "kmboxNet.h"
#include "pid.h"

// 这些参数由 main.cpp 定义，ImGuiUI 可实时修改
extern float fc_h1lo, fc_h1hi, fc_s1lo, fc_s1hi, fc_v1lo, fc_v1hi;
extern float fc_h2lo, fc_h2hi, fc_s2lo, fc_s2hi, fc_v2lo, fc_v2hi;
extern int   fc_roi_x, fc_roi_y, fc_roi_w, fc_roi_h;
extern float fc_aim_scale;
extern int   fc_min_area;
extern int   g_fcRadius;

static int lostFrames = 0;

// screenW / screenH: 由调用方传入有效屏幕分辨率
inline void FindColor(const cv::Mat& src, float screenW, float screenH) {
    std::vector<cv::Point2f> centers;
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask1, mask2, mask;
    // 使用 UI 可调的 HSV 范围
    cv::inRange(hsv,
        cv::Scalar(fc_h1lo, fc_s1lo, fc_v1lo),
        cv::Scalar(fc_h1hi, fc_s1hi, fc_v1hi), mask1);
    cv::inRange(hsv,
        cv::Scalar(fc_h2lo, fc_s2lo, fc_v2lo),
        cv::Scalar(fc_h2hi, fc_s2hi, fc_v2hi), mask2);
    mask = mask1 | mask2;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < fc_min_area) continue;
        cv::Moments m = cv::moments(c);
        if (m.m00 == 0) continue;
        centers.push_back({
            static_cast<float>(m.m10 / m.m00),
            static_cast<float>(m.m01 / m.m00)
            });
    }

    // ROI 中心（相对于 ROI 本身）
    cv::Point2f target(src.cols * 0.5f, src.rows * 0.5f);
    cv::Point2f nearest(-1.f, -1.f);
    float minDistSq = std::numeric_limits<float>::max();
    for (const auto& pt : centers) {
        float dx = pt.x - target.x, dy = pt.y - target.y;
        float dSq = dx * dx + dy * dy;
        if (dSq < minDistSq) { minDistSq = dSq; nearest = pt; }
    }

    if (nearest.x < 0) {
        lostFrames++;
        if (lostFrames >= 5) CenterToScreen(screenW * 0.5f, screenH * 0.5f);
        return;
    }
    lostFrames = 0;

    // 将 ROI 内坐标映射回屏幕（ROI 偏移由调用方决定，这里补回全局中心）
    // ROI 起点 = (screenCenterX + fc_roi_x, screenCenterY + fc_roi_y)
    // 实际屏幕坐标 = ROI起点 + nearest位置
    float screenCX = screenW * 0.5f;
    float screenCY = screenH * 0.5f;
    float roiOriginX = screenCX + fc_roi_x;
    float roiOriginY = screenCY + fc_roi_y;
    float rawX = roiOriginX + nearest.x;
    float rawY = roiOriginY + nearest.y;

    // 对偏移应用缩放（使瞄准更平滑）
    float dx = rawX - screenCX, dy = rawY - screenCY;
    if (abs(dx) < 3) dx = 0;
    if (abs(dy) < 3) dy = 0;
    float finalX = screenCX + dx * fc_aim_scale;
    float finalY = screenCY + dy * fc_aim_scale;

    if (kmNet_monitor_mouse_right()) {
        CenterToScreen(finalX, finalY);
    }
    else {
        CenterToScreen(screenCX, screenCY);
    }
}
