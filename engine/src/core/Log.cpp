#include <fstream>

#include "../../include/core/Log.h"

using namespace PhysicsEngine;

size_t Log::maxMessageCount = 100;
std::queue<std::string> Log::messages;

void Log::info(const char *format, ...)
{
    if (messages.size() >= maxMessageCount)
    {
        return;
    }

    va_list args;
    va_start(args, format);

    std::string message = convertToString(Level::Info, format, args);

    va_end(args);

    messages.push(message);
    //messages.push("[Info]: " + message);
}

void Log::warn(const char *format, ...)
{
    if (messages.size() >= maxMessageCount)
    {
        return;
    }

    va_list args;
    va_start(args, format);

    std::string message = convertToString(Level::Warn, format, args);

    va_end(args);

    messages.push("[Warn]: " + message);
}

void Log::error(const char *format, ...)
{
    if (messages.size() >= maxMessageCount)
    {
        return;
    }

    va_list args;
    va_start(args, format);

    std::string message = convertToString(Level::Error, format, args);

    va_end(args);

    messages.push("[Error]: " + message);
}

void Log::clear()
{
    while (!messages.empty())
    {
        messages.pop();
    }
}

std::queue<std::string> Log::getMessages()
{
    return messages;
}

//std::string Log::convertToString(Level level, const char *format, va_list args)
//{
//    std::string message = "";
//
//    const char *c = format;
//    while (*c != '\0')
//    {
//        if (*c != '%')
//        {
//            message += *c;
//            c++;
//        }
//        else
//        {
//            c++;
//            if (*c == '\0')
//            {
//                break;
//            }
//
//            switch (*c)
//            {
//            case 'd':
//                message += std::to_string(va_arg(args, int));
//                break;
//            case 'i':
//                message += std::to_string(va_arg(args, int));
//                break;
//            case 'u':
//                message += std::to_string(va_arg(args, int));
//                break;
//            case 'c':
//                message += static_cast<char>(va_arg(args, int));
//                break;
//            case 'f':
//                message += std::to_string(va_arg(args, double));
//                break;
//            case 'p':
//                char text[255];
//                sprintf(text, "%p\n", va_arg(args, void *));
//                int t = 0;
//                while (t <= 255 && text[t] != '\n')
//                {
//                    message += text[t];
//                    t++;
//                }
//                break;
//            }
//
//            c++;
//        }
//    }
//
//    return message;
//}

std::string Log::convertToString(Level level, const char* format, va_list args)
{
    size_t i = 0;
    char buffer[16384];

    switch (level)
    {
    case Level::Info:
        strncpy(buffer, "[Info]: ", 8);
        i += 8;
        break;
    case Level::Warn:
        strncpy(buffer, "[Warn]: ", 8);
        i += 8;
        break;
    case Level::Error:
        strncpy(buffer, "[Error]: ", 9);
        i += 9;
        break;
    }

    const char* c = format;
    while (*c != '\0')
    {
        if (*c != '%')
        {
            buffer[i] = *c;
            i++;
            c++;
        }
        else
        {
            c++;
            if (*c == '\0')
            {
                break;
            }

            std::string temp;
            switch (*c)
            {
            case 'd':
            case 'i':
            case 'u':
                temp = std::to_string(va_arg(args, int));
                break;
            case 'c':
                temp = static_cast<char>(va_arg(args, int));
                break;
            case 'f':
                temp = std::to_string(va_arg(args, double));
                break;
            case 'p':
                /*char text[255];
                sprintf(text, "%p\n", va_arg(args, void*));
                int t = 0;
                while (t <= 255 && text[t] != '\n')
                {
                    message += text[t];
                    t++;
                }*/
                break;
            }

            strncpy(&buffer[i], temp.c_str(), temp.length());
            c++;
            i += temp.length();
        }
    }

    buffer[i] = '\0';

    return std::string(buffer);
}