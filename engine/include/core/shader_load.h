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

bool shader_load(const std::string &sourceFilepath, shader_data &data)
{
    // extract all contents from shader source filepath
    std::ifstream sourceFile(sourceFilepath, std::ios::in);
    std::ostringstream sourceContents;
    if (sourceFile.is_open())
    {
        sourceContents << sourceFile.rdbuf();
        sourceFile.close();
    }
    else
    {
        return false;
    }

    std::string contents = sourceContents.str();

    const std::string VERTEX = "#vertex\n";
    const std::string FRAGMENT = "#fragment\n";
    const std::string GEOMETRY = "#geometry\n";

    size_t vertex_start_index = contents.find(VERTEX);
    size_t fragment_start_index = contents.find(FRAGMENT);
    size_t geometry_start_index = contents.find(GEOMETRY);

    bool has_vertex_shader = vertex_start_index != std::string::npos;
    bool has_fragment_shader = fragment_start_index != std::string::npos;
    bool has_geometry_shader = geometry_start_index != std::string::npos;

    if (!has_vertex_shader){ return false; }
    if (!has_fragment_shader){ return false; }

    if (has_geometry_shader)
    {
        if (vertex_start_index < fragment_start_index)
        {
        
        }
    }
    else
    {
        // listed in source as vertex then fragment
        if (vertex_start_index < fragment_start_index)
        {
            data.mVertexShader = contents.substr(vertex_start_index + VERTEX.length(),
                                                 fragment_start_index - (vertex_start_index + VERTEX.length()));
            data.mFragmentShader = contents.substr(fragment_start_index + FRAGMENT.length());
        }
        // listed in source as fragment then vertex
        else
        {
            data.mFragmentShader = contents.substr(fragment_start_index + FRAGMENT.length(),
                                                   vertex_start_index - (fragment_start_index + FRAGMENT.length()));
            data.mVertexShader = contents.substr(vertex_start_index + VERTEX.length());
        }

        data.mGeometryShader = "";
    }

    return true;
}

} // namespace PhysicsEngine

#endif
