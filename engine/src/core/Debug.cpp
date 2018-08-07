#include <iostream>
#include <fstream>

#include "../../include/core/Debug.h"

using namespace PhysicsEngine;

bool Debug::debugOn = true;
std::ofstream Debug::logFile;
std::vector<DebugLine> Debug::debugLines;

void Debug::LogInfo(std::string message)
{
	logFile.open("log.txt");
	logFile << "[INFO] " << message << "\n";
	logFile.close();
}

void Debug::LogWarning(std::string message)
{
	logFile.open("log.txt");
	std::cout << "[WARN] " << message << std::endl;
	logFile.close();
}

void Debug::LogError(std::string message)
{
	logFile.open("log.txt");
	std::cerr << "[ERROR] " << message << std::endl;
	logFile.close();
}

void Debug::drawLine(glm::vec3 start, glm::vec3 end)// , Color color)
{
	debugLines.push_back(DebugLine(start, end));
}

void Debug::drawRay(glm::vec3 start, glm::vec3 direction, Color color)
{

}