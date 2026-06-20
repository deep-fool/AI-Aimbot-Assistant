#include "ScreenCapture.h"
#include <chrono>

ScreenCapture::ScreenCapture()
    : width(0)
    , height(0)
    , channels(3)
    , captureTime(0.0f)
    , initialized(false)
    , method(1) {
}

ScreenCapture::~ScreenCapture() {
    Release();
}

bool ScreenCapture::Init(int width, int height, int mode) {
    Release();
    method = mode;

    this->width = width;
    this->height = height;

    if (method == 0) {
        gdiCapture = std::make_unique<GDIScreenCapture>();
        if (gdiCapture->Initialize(width, height)) {
            initialized = true;
            return true;
        }
        gdiCapture.reset();
    }
    else if (method == 1) {
        dxCapture = std::make_unique<DXScreenCapture>();
        if (dxCapture->Initialize(width, height)) {
            initialized = true;
            return true;
        }
        dxCapture.reset();
    }
    else if (method == 2) {
        wgcCapture = std::make_unique<WGCScreenCapture>();
        if (wgcCapture->Initialize(width, height)) {
            initialized = true;
            return true;
        }
        wgcCapture.reset();
    }

    return false;
}

bool ScreenCapture::SetWindow(HWND hwnd) {
    if (!initialized) return false;

    switch (method) {
    case 0:
        return gdiCapture ? gdiCapture->SetWindow(hwnd) : false;
    case 1:
        return false;
    case 2:
        return wgcCapture ? wgcCapture->SetWindow(hwnd) : false;
    default:
        return false;
    }
}

bool ScreenCapture::SetRegion(int x, int y, int width, int height) {
    if (!initialized) return false;

    bool success = false;

    switch (method) {
    case 0:
        success = gdiCapture ? gdiCapture->SetRegion(x, y, width, height) : false;
        break;
    case 1:
        success = dxCapture ? dxCapture->SetRegion(x, y, width, height) : false;
        break;
    case 2:
        success = wgcCapture ? wgcCapture->SetRegion(x, y, width, height) : false;
        break;
    default:
        return false;
    }

    if (success) {
        this->width = width;
        this->height = height;
    }

    return success;
}

void ScreenCapture::Reset() {
    Release();
    initialized = false;
    width = 0;
    height = 0;
    captureTime = 0.0f;
}

std::vector<uint8_t> ScreenCapture::Capture() {
    if (!initialized) return std::vector<uint8_t>();

    switch (method) {
    case 0:
        return gdiCapture ? gdiCapture->CaptureBGR() : std::vector<uint8_t>();
    case 1:
        return dxCapture ? dxCapture->CaptureBGR() : std::vector<uint8_t>();
    case 2:
        return wgcCapture ? wgcCapture->CaptureBGR() : std::vector<uint8_t>();
    default:
        return std::vector<uint8_t>();
    }
}

bool ScreenCapture::Capture(uint8_t* buffer, size_t bufferSize) {
    if (!initialized || !buffer) return false;

    switch (method) {
    case 0:
        return gdiCapture ? gdiCapture->CaptureBGR(buffer, bufferSize) : false;
    case 1:
        return dxCapture ? dxCapture->CaptureBGR(buffer, bufferSize) : false;
    case 2:
        return wgcCapture ? wgcCapture->CaptureBGR(buffer, bufferSize) : false;
    default:
        return false;
    }
}

void ScreenCapture::Release() {
    if (gdiCapture) {
        gdiCapture->Release();
        gdiCapture.reset();
    }

    if (dxCapture) {
        dxCapture->Release();
        dxCapture.reset();
    }

    if (wgcCapture) {
        wgcCapture->Release();
        wgcCapture.reset();
    }
}
