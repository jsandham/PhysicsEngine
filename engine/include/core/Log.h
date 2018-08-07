#ifndef __LOG_H__
#define __LOG_H__

#include <string>
#include <cstdarg>

namespace PhysicsEngine
{
	class Log
	{
		private:
			static std::string convertToString(const char* format, va_list args);

		public:
			static void Info(const char* format, ...);
			static void Warn(const char* format, ...);
			static void Error(const char* format, ...);
	};
}

#endif