     #pragma once

#ifdef BUILD_DLL
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

#include <cstdint>

extern "C" {
    API void*    SCap_Create();
    API void     SCap_Destroy(void* handle);
    API bool     SCap_Init(void* handle, int w, int h, int mode);
    API bool     SCap_SetRegion(void* handle, int x, int y, int w, int h);
    API bool     SCap_Capture(void* handle, uint8_t* buf, int bufSize);
    API int      SCap_GetWidth(void* handle);
    API int      SCap_GetHeight(void* handle);
    API int      SCap_GetChannels(void* handle);
    API void     SCap_Reset(void* handle);
}
