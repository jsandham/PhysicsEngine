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

bool shader_load(const std::string &vsFilepath, const std::string &fsFilepath, const std::string &gsFilepath,
                 shader_data &data)
{
    // vertex shader
    std::ifstream vsFile(vsFilepath, std::ios::in);
    std::ostringstream vsContents;
    if (vsFile.is_open())
    {
        vsContents << vsFile.rdbuf();
        data.mVertexShader = vsContents.str();
        vsFile.close();
    }
    else
    {
        return false;
    }

    // fragment shader
    std::ifstream fsFile(fsFilepath, std::ios::in);
    std::ostringstream fsContents;
    if (fsFile.is_open())
    {
        fsContents << fsFile.rdbuf();
        data.mFragmentShader = fsContents.str();
        fsFile.close();
    }
    else
    {
        return false;
    }

    // geometry shader
    if (!gsFilepath.empty())
    {
        std::ifstream gsFile(gsFilepath, std::ios::in);
        std::ostringstream gsContents;
        if (gsFile.is_open())
        {
            gsContents << gsFile.rdbuf();
            data.mGeometryShader = gsContents.str();
            gsFile.close();
        }
        else
        {
            return false;
        }
    }

    return true;
}

} // namespace PhysicsEngine

#endif
