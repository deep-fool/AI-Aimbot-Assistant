#ifndef SCREENCAPTURE_H
#define SCREENCAPTURE_H

#include "GDIScreenCapture.h"
#include "DXScreenCapture.h"
#include "WGCScreenCapture.h"
#include <vector>
#include <cstdint>
#include <memory>
#include <Windows.h>

class ScreenCapture {
public:
    ScreenCapture();
    ~ScreenCapture();

    bool Init(int width, int height, int mode);
    bool SetWindow(HWND hwnd);
    bool SetRegion(int x, int y, int width, int height);
    void Reset();
    std::vector<uint8_t> Capture();
    bool Capture(uint8_t* buffer, size_t bufferSize);

    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
    int GetChannels() const { return channels; }

private:
    void Release();

    std::unique_ptr<GDIScreenCapture> gdiCapture;
    std::unique_ptr<DXScreenCapture> dxCapture;
    std::unique_ptr<WGCScreenCapture> wgcCapture;

    int width;
    int height;
    int channels;
    float captureTime;
    bool initialized;
    int method; // 0=GDI, 1=DXGI, 2=WGC
};

#endif
