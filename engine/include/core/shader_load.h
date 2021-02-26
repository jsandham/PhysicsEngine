#ifndef SHADER_LOADER_H__
#define SHADER_LOADER_H__

#include <fstream>
#include <sstream>
#include <string>

namespace PhysicsEngine
{
typedef struct shader_data
{
    std::string mVertexShader;
    std::string mGeometryShader;
    std::string mFragmentShader;
} shader_data;

bool shader_load(const std::string &filepath, shader_data &data)
{
    std::ifstream file(filepath, std::ios::in);
    std::ostringstream contents;
    if (file.is_open())
    {
        contents << file.rdbuf();
        file.close();
    }
    else
    {
        return false;
    }

    std::string shaderContent = contents.str();

    const std::string vertexTag = "VERTEX:";
    const std::string geometryTag = "GEOMETRY:";
    const std::string fragmentTag = "FRAGMENT:";

    size_t startOfVertexTag = shaderContent.find(vertexTag, 0);
    size_t startOfGeometryTag = shaderContent.find(geometryTag, 0);
    size_t startOfFragmentTag = shaderContent.find(fragmentTag, 0);

    if (startOfVertexTag == std::string::npos || startOfFragmentTag == std::string::npos)
    {
        return false;
    }

    std::string vertexShader, geometryShader, fragmentShader;

    if (startOfGeometryTag == std::string::npos)
    {
        vertexShader =
            shaderContent.substr(startOfVertexTag + vertexTag.length(), startOfFragmentTag - vertexTag.length());
        geometryShader = "";
        fragmentShader = shaderContent.substr(startOfFragmentTag + fragmentTag.length(), shaderContent.length());
    }
    else
    {
        vertexShader =
            shaderContent.substr(startOfVertexTag + vertexTag.length(), startOfGeometryTag - vertexTag.length());
        geometryShader =
            shaderContent.substr(startOfGeometryTag + geometryTag.length(), startOfFragmentTag - geometryTag.length());
        fragmentShader = shaderContent.substr(startOfFragmentTag + fragmentTag.length(), shaderContent.length());
    }

    // trim left
    size_t firstNotOfIndex;
    firstNotOfIndex = vertexShader.find_first_not_of("\n");
    if (firstNotOfIndex != std::string::npos)
    {
        vertexShader = vertexShader.substr(firstNotOfIndex);
    }

    firstNotOfIndex = geometryShader.find_first_not_of("\n");
    if (firstNotOfIndex != std::string::npos)
    {
        geometryShader = geometryShader.substr(firstNotOfIndex);
    }

    firstNotOfIndex = fragmentShader.find_first_not_of("\n");
    if (firstNotOfIndex != std::string::npos)
    {
        fragmentShader = fragmentShader.substr(firstNotOfIndex);
    }

    // trim right
    size_t lastNotOfIndex;
    lastNotOfIndex = vertexShader.find_last_not_of("\n");
    if (lastNotOfIndex != std::string::npos)
    {
        vertexShader.erase(lastNotOfIndex + 1);
    }

    lastNotOfIndex = geometryShader.find_last_not_of("\n");
    if (lastNotOfIndex != std::string::npos)
    {
        geometryShader.erase(lastNotOfIndex + 1);
    }

    lastNotOfIndex = fragmentShader.find_last_not_of("\n");
    if (lastNotOfIndex != std::string::npos)
    {
        fragmentShader.erase(lastNotOfIndex + 1);
    }

    data.mVertexShader = vertexShader;
    data.mGeometryShader = geometryShader;
    data.mFragmentShader = fragmentShader;

    return true;
}
} // namespace PhysicsEngine

#endif
