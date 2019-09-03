#ifndef __LOG_H__
#define __LOG_H__

#include <string>
#include <cstdarg>
#include <queue>

namespace PhysicsEngine
{
	class Log
	{
		private:
			static std::queue<std::string> messages;
			static std::string convertToString(const char* format, va_list args);

		public:
			static int maxMessageCount;

			static void info(const char* format, ...);
			static void warn(const char* format, ...);
			static void error(const char* format, ...);

			static void clear();
			static std::queue<std::string> getMessages();
	};
}

#endif