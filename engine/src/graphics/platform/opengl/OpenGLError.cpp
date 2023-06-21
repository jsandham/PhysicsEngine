#include <GL/glew.h>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <string>

#include "../../../../include/graphics/platform/opengl/OpenGLError.h"
#include "../../../../include/core/Log.h"

using namespace PhysicsEngine;

static const std::string INVALID_ENUM = "An unacceptable value is specified for an enumerated argument";
static const std::string INVALID_VALUE = "A numeric argument is out of range";
static const std::string INVALID_OPERATION = "The specified operation is not allowed in the current state";
static const std::string INVALID_FRAMEBUFFER_OPERATION = "The framebuffer object is not complete";
static const std::string OUT_OF_MEMORY = "There is not enough memory left to execute the command";
static const std::string STACK_UNDERFLOW =
    "An attempt has been made to perform an operation that would cause an internal stack to underflow";
static const std::string STACK_OVERFLOW =
    "An attempt has been made to perform an operation that would cause an internal stack to overflow";
static const std::string UNKNOWN_ERROR = "Unknown error";
static const std::string FRAMEBUFFER_UNDEFINED = "The current FBO binding is 0 but no default framebuffer exists";
static const std::string FRAMEBUFFER_INCOMPLETE_ATTACHMENT = "One of the buffers enabled for rendering is incomplete";
static const std::string FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT =
    "No buffers are attached to the FBO and it is not configured for rendering without attachments";
static const std::string FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER =
    "Not all attachments enabled via glDrawBuffers exists in framebuffer";
static const std::string FRAMEBUFFER_INCOMPLETE_READ_BUFFER =
    "Not all buffers specified via glReadBuffer exists in framebuffer";
static const std::string FRAMEBUFFER_UNSUPPORTED = "The combination of internal buffer formats is unsupported";
static const std::string FRAMEBUFFER_INCOMPLETE_MULTISAMPLE =
    "The number of samples for each attachment is not the same";
static const std::string FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS =
    "Not all color attachments are layered textures or bound to the same target";
static const std::string UNKNOWN_FRAMEBUFFER_ERROR = "Unknown framebuffer status error";

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

void PhysicsEngine::checkError(int line, const char* file)
{
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR)
    {
        switch (error)
        {
        case GL_INVALID_ENUM:
            LOG_ERROR(INVALID_ENUM, std::to_string(line), std::string(file));
            break;
        case GL_INVALID_VALUE:
            LOG_ERROR(INVALID_VALUE, std::to_string(line), std::string(file));
            break;
        case GL_INVALID_OPERATION:
            LOG_ERROR(INVALID_OPERATION, std::to_string(line), std::string(file));
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            LOG_ERROR(INVALID_FRAMEBUFFER_OPERATION, std::to_string(line), std::string(file));
            break;
        case GL_OUT_OF_MEMORY:
            LOG_ERROR(OUT_OF_MEMORY, std::to_string(line), std::string(file));
            break;
        case GL_STACK_UNDERFLOW:
            LOG_ERROR(STACK_UNDERFLOW, std::to_string(line), std::string(file));
            break;
        case GL_STACK_OVERFLOW:
            LOG_ERROR(STACK_OVERFLOW, std::to_string(line), std::string(file));
            break;
        default:
            LOG_ERROR(UNKNOWN_ERROR, std::to_string(line), std::string(file));
            break;
        }
    }
}

void PhysicsEngine::checkFrambufferError(int line, const char* file)
{
    GLenum framebufferStatus = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    if (framebufferStatus != GL_FRAMEBUFFER_COMPLETE)
    {
        switch (framebufferStatus)
        {
        case GL_FRAMEBUFFER_UNDEFINED:
            LOG_ERROR(FRAMEBUFFER_UNDEFINED, std::to_string(line), std::string(file));
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            LOG_ERROR(FRAMEBUFFER_INCOMPLETE_ATTACHMENT, std::to_string(line), std::string(file));
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            LOG_ERROR(FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT, std::to_string(line), std::string(file));
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            LOG_ERROR(FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER, std::to_string(line), std::string(file));
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            LOG_ERROR(FRAMEBUFFER_INCOMPLETE_READ_BUFFER, std::to_string(line), std::string(file));
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED:
            LOG_ERROR(FRAMEBUFFER_UNSUPPORTED, std::to_string(line), std::string(file));
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            LOG_ERROR(FRAMEBUFFER_INCOMPLETE_MULTISAMPLE, std::to_string(line), std::string(file));
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
            LOG_ERROR(FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS, std::to_string(line), std::string(file));
            break;
        default:
            LOG_ERROR(UNKNOWN_FRAMEBUFFER_ERROR, std::to_string(line), std::string(file));
            break;
        }
    }
}