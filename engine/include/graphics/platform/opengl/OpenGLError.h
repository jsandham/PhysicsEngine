#ifndef OPENGL_ERROR_H__
#define OPENGL_ERROR_H__

namespace PhysicsEngine
{
void checkError(int line, const char *file);
void checkFrambufferError(int line, const char *file);

#define CHECK_ERROR_IMPL(ROUTINE, LINE, FILE)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        ROUTINE;                                                                                                       \
        checkError(LINE, FILE);                                                                                        \
    } while (0)

#define CHECK_ERROR(ROUTINE) CHECK_ERROR_IMPL(ROUTINE, __LINE__, __FILE__)
} // namespace PhysicsEngine

#endif