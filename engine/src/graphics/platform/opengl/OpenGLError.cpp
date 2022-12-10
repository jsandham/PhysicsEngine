#include <GL/glew.h>
#include <algorithm>
#include <assert.h>
#include <iostream>

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

void PhysicsEngine::logError(const std::string &error, const std::string &line, const std::string &file)
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

void PhysicsEngine::checkError(const std::string &line, const std::string &file)
{
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR)
    {
        switch (error)
        {
        case GL_INVALID_ENUM:
            LOG_ERROR(INVALID_ENUM, line, file);
            break;
        case GL_INVALID_VALUE:
            LOG_ERROR(INVALID_VALUE, line, file);
            break;
        case GL_INVALID_OPERATION:
            LOG_ERROR(INVALID_OPERATION, line, file);
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            LOG_ERROR(INVALID_FRAMEBUFFER_OPERATION, line, file);
            break;
        case GL_OUT_OF_MEMORY:
            LOG_ERROR(OUT_OF_MEMORY, line, file);
            break;
        case GL_STACK_UNDERFLOW:
            LOG_ERROR(STACK_UNDERFLOW, line, file);
            break;
        case GL_STACK_OVERFLOW:
            LOG_ERROR(STACK_OVERFLOW, line, file);
            break;
        default:
            LOG_ERROR(UNKNOWN_ERROR, line, file);
            break;
        }
    }
}

void PhysicsEngine::checkFrambufferError(const std::string &line, const std::string &file)
{
    GLenum framebufferStatus = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    if (framebufferStatus != GL_FRAMEBUFFER_COMPLETE)
    {
        switch (framebufferStatus)
        {
        case GL_FRAMEBUFFER_UNDEFINED:
            LOG_ERROR(FRAMEBUFFER_UNDEFINED, line, file);
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            LOG_ERROR(FRAMEBUFFER_INCOMPLETE_ATTACHMENT, line, file);
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            LOG_ERROR(FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT, line, file);
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            LOG_ERROR(FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER, line, file);
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            LOG_ERROR(FRAMEBUFFER_INCOMPLETE_READ_BUFFER, line, file);
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED:
            LOG_ERROR(FRAMEBUFFER_UNSUPPORTED, line, file);
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            LOG_ERROR(FRAMEBUFFER_INCOMPLETE_MULTISAMPLE, line, file);
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
            LOG_ERROR(FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS, line, file);
            break;
        default:
            LOG_ERROR(UNKNOWN_FRAMEBUFFER_ERROR, line, file);
            break;
        }
    }
}