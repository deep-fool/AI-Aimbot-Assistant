#define BUILD_DLL
#include "exports.h"
#include "ScreenCapture.h"
#pragma execution_character_set("utf-8")
extern "C" {

API void* SCap_Create() { return new (std::nothrow) ScreenCapture(); }

API void SCap_Destroy(void* handle) { delete static_cast<ScreenCapture*>(handle); }

API bool SCap_Init(void* handle, int w, int h, int mode) {
    if (!handle) return false;
    return static_cast<ScreenCapture*>(handle)->Init(w, h, mode);
}

API bool SCap_SetRegion(void* handle, int x, int y, int w, int h) {
    if (!handle) return false;
    return static_cast<ScreenCapture*>(handle)->SetRegion(x, y, w, h);
}

API bool SCap_Capture(void* handle, uint8_t* buf, int bufSize) {
    if (!handle || !buf || bufSize <= 0) return false;
    return static_cast<ScreenCapture*>(handle)->Capture(buf, (size_t)bufSize);
}

API int SCap_GetWidth(void* handle) {
    if (!handle) return 0;
    return static_cast<ScreenCapture*>(handle)->GetWidth();
}

API int SCap_GetHeight(void* handle) {
    if (!handle) return 0;
    return static_cast<ScreenCapture*>(handle)->GetHeight();
}

API int SCap_GetChannels(void* handle) {
    if (!handle) return 0;
    return static_cast<ScreenCapture*>(handle)->GetChannels();
}

API void SCap_Reset(void* handle) {
    if (handle) static_cast<ScreenCapture*>(handle)->Reset();
}

}
