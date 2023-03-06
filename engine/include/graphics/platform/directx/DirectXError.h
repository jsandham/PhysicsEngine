#ifndef DIRECTX_ERROR_H__
#define DIRECTX_ERROR_H__

#include <d3d11.h>
#include <d3dcompiler.h>

#include <string>

namespace PhysicsEngine
{
    void checkError(HRESULT result, const std::string &line, const std::string &file);

    #define CHECK_ERROR_IMPL(ROUTINE, LINE, FILE)                                     \
        do                                                                            \
        {                                                                             \
            HRESULT hr = ROUTINE;                                                     \
            checkError(hr, LINE, FILE);                                               \
        } while (0)

    #define CHECK_ERROR(ROUTINE) CHECK_ERROR_IMPL(ROUTINE, std::to_string(__LINE__), std::string(__FILE__))
} // namespace PhysicsEngine

#endif