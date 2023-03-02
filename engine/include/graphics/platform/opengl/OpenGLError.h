#ifndef OPENGL_ERROR_H__
#define OPENGL_ERROR_H__ 

#include <string>

namespace PhysicsEngine
{
    void logError(const std::string &error, const std::string &line, const std::string &file);

    #define LOG_ERROR(ERROR, LINE, FILE) logError(ERROR, LINE, FILE);

    void checkError(const std::string &line, const std::string &file);
    void checkFrambufferError(const std::string &line, const std::string &file);

    /*#define CHECK_ERROR_IMPL(ROUTINE, LINE, FILE)                                                                          \
        do                                                                                                                 \
        {                                                                                                                  \
            ROUTINE;                                                                                                       \
            checkError(LINE, FILE);                                                                                        \
        } while (0)*/
#define CHECK_ERROR_IMPL(ROUTINE, LINE, FILE)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        ROUTINE;                                                                                                       \
    } while (0)

    #define CHECK_ERROR(ROUTINE) CHECK_ERROR_IMPL(ROUTINE, std::to_string(__LINE__), std::string(__FILE__))
}

#endif