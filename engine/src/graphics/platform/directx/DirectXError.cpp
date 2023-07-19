#include <algorithm>
#include <assert.h>
#include <iostream>

#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

using namespace PhysicsEngine;

static const std::string ERROR_FILE_NOT_FOUND1 = "ERROR_FILE_NOT_FOUND";
static const std::string ERROR_TOO_MANY_UNIQUE_STATE_OBJECTS = "ERROR_TOO_MANY_UNIQUE_STATE_OBJECTS";
static const std::string ERROR_TOO_MANY_UNIQUE_VIEW_OBJECTS = "ERROR_TOO_MANY_UNIQUE_VIEW_OBJECTS";
static const std::string ERROR_DEFERRED_CONTEXT_MAP_WITHOUT_INITIAL_DISCARD =
    "ERROR_DEFERRED_CONTEXT_MAP_WITHOUT_INITIAL_DISCARD";
static const std::string ERROR_INVALID_CALL = "ERROR_INVALID_CALL";
static const std::string ERROR_WAS_STILL_DRAWING = "ERROR_WAS_STILL_DRAWING";
static const std::string FAIL = "FAIL";
static const std::string INVALIDARG = "INVALIDARG";
static const std::string OUTOFMEMORY = "OUTOFMEMORY";
static const std::string NOTIMPL = "NOTIMPL";
static const std::string FALSE1 = "FALSE";

static void logError(const std::string &error, const std::string &line, const std::string &file)
{
    char errorBuffer[512];
    size_t i = 1;
    errorBuffer[0] = '(';
    strncpy(errorBuffer + i, error.c_str(), error.length());
    i += error.length();
    strncpy(errorBuffer + i, ") line: ", 8);
    i += 8;
    strncpy(errorBuffer + i, line.c_str(), line.length());
    i += line.length();
    strncpy(errorBuffer + i, " file: ", 7);
    i += 7;
    strncpy(errorBuffer + i, file.c_str(), file.length());
    i += file.length();
    errorBuffer[i] = '\n';
    errorBuffer[i + 1] = '\0';
    Log::error(errorBuffer);
}

#define LOG_ERROR(ERROR, LINE, FILE) logError(ERROR, LINE, FILE);

void PhysicsEngine::checkError(HRESULT result, const std::string &line, const std::string &file)
{
    if (result == S_OK)
        return;

    switch (result)
    {
    case D3D11_ERROR_FILE_NOT_FOUND:
        LOG_ERROR(ERROR_FILE_NOT_FOUND1, line, file);
        break;
    case D3D11_ERROR_TOO_MANY_UNIQUE_STATE_OBJECTS:
        LOG_ERROR(ERROR_TOO_MANY_UNIQUE_STATE_OBJECTS, line, file);
        break;
    case D3D11_ERROR_TOO_MANY_UNIQUE_VIEW_OBJECTS:
        LOG_ERROR(ERROR_TOO_MANY_UNIQUE_VIEW_OBJECTS, line, file);
        break;
    case D3D11_ERROR_DEFERRED_CONTEXT_MAP_WITHOUT_INITIAL_DISCARD:
        LOG_ERROR(ERROR_DEFERRED_CONTEXT_MAP_WITHOUT_INITIAL_DISCARD, line, file);
        break;
    case DXGI_ERROR_INVALID_CALL:
        LOG_ERROR(ERROR_INVALID_CALL, line, file);
        break;
    case DXGI_ERROR_WAS_STILL_DRAWING:
        LOG_ERROR(ERROR_WAS_STILL_DRAWING, line, file);
        break;
    case E_FAIL:
        LOG_ERROR(FAIL, line, file);
        break;
    case E_INVALIDARG:
        LOG_ERROR(INVALIDARG, line, file);
        break;
    case E_OUTOFMEMORY:
        LOG_ERROR(OUTOFMEMORY, line, file);
        break;
    case E_NOTIMPL:
        LOG_ERROR(NOTIMPL, line, file);
        break;
    case S_FALSE:
        LOG_ERROR(FALSE1, line, file);
        break;
    }
}