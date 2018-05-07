#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <iostream>
#include <fstream>
#include <string>

#include "../graphics/Color.h"
#include "../graphics/DebugLine.h"

#include "../glm/glm.hpp"

#define DEBUG 0

namespace PhysicsEngine
{
	class Debug
	{
		public:
			static bool debugOn;
			static std::ofstream logFile;
			static std::vector<DebugLine> debugLines;

			static void LogInfo(std::string message);
			static void LogWarning(std::string message);
			static void LogError(std::string message);

			static void drawLine(glm::vec3 start, glm::vec3 end);// , Color color);
			static void drawRay(glm::vec3 start, glm::vec3 direction, Color color);
	};
}

#endif