#include <fstream>

#include "../../include/core/Log.h"

using namespace PhysicsEngine;

std::string Log::convertToString(const char* format, va_list args)
{
	std::string message = "";

	const char* c = format;
	while (*c != '\0'){
		if (*c != '%'){
			message += *c;
			c++;
		}
		else{
			c++;
			if (*c == '\0'){ break; }

			switch (*c)
			{
			case 'd':
				message += std::to_string(va_arg(args, int));
				break;
			case 'i':
				message += std::to_string(va_arg(args, int));
				break;
			case 'u':
				message += std::to_string(va_arg(args, int));
				break;
			case 'c':
				message += static_cast<char>(va_arg(args, int));
				break;
			case 'f':
				message += std::to_string(va_arg(args, double));
				break;
			case 'p':
				char text[255];
				sprintf(text, "%p\n", va_arg(args, void*));
				int t = 0;
				while(t <= 255 && text[t] != '\n'){
					message += text[t];
					t++;
				}
				break;
			}

			c++;
		}
	}

	return message;
}

void Log::Info(const char* format, ...)
{
	va_list args;
	va_start(args, format);

	std::string message = convertToString(format, args);

	va_end(args);

	std::ofstream logFile("log.txt", std::ios_base::out | std::ios_base::app);
	logFile << "[Info]: " << message << std::endl;
}

void Log::Warn(const char* format, ...)
{
	va_list args;
	va_start(args, format);

	std::string message = convertToString(format, args);

	va_end(args);

	std::ofstream logFile("log.txt", std::ios_base::out | std::ios_base::app);
	logFile << "[Warn]: " << message << std::endl;
}

void Log::Error(const char* format, ...)
{
	va_list args;
	va_start(args, format);

	std::string message = convertToString(format, args);

	va_end(args);

	std::ofstream logFile("log.txt", std::ios_base::out | std::ios_base::app);
	logFile << "[Error]: " << message << std::endl;
}