#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stack>
#include <unordered_map>
#include <filesystem>

struct File
{
    std::string path;
    char *buffer;
    size_t bufferLength;
};

char *readEntireFileIntoBuffer(const std::string &filename, size_t &bufferLength)
{
    std::ifstream in;
    in.open(filename, std::ifstream::in);

    if (!in.is_open())
    {
        return nullptr;
    }

    std::stringstream ss;
    ss << in.rdbuf();
    in.close();

    std::string str = ss.str();
    int length = str.length();

    char *buffer = (char *)malloc((length + 1) * sizeof(char));
    memcpy(buffer, str.c_str(), length * sizeof(char));

    buffer[length] = '\0';

    bufferLength = length + 1;

    // std::cout << "file contents" << std::endl;
    // std::cout << str << std::endl;

    return buffer;
}

std::vector<File> loadFilesInDirectory(const std::string& directoryPath)
{
    if (!std::filesystem::exists(directoryPath))
    {
        return std::vector<File>();
    }

    std::vector<File> files;

    std::error_code error_code;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directoryPath, error_code))
    {
        if (std::filesystem::is_regular_file(entry, error_code) && entry.path().extension() == ".hlsl")
        {
            size_t bufferLength;
            char* buffer = readEntireFileIntoBuffer(entry.path().string(), bufferLength);
            if (buffer != nullptr)
            {
                File file;
                file.path = entry.path().string();
                file.buffer = buffer;
                file.bufferLength = bufferLength;

                files.push_back(file);

                std::cout << "File path: " << file.path << std::endl;
            }
        }
    }

    return files;
}

void write_header(const std::string& name, std::ofstream& out)
{
    out << "//***************************************\n";
    out << "// THIS IS A GENERATED FILE. DO NOT EDIT.\n";
    out << "//***************************************\n";
    out << "#include <string>\n";
    out << "#include \"hlsl_shaders.h\"\n";
    out << ("using namespace " + name + ";\n");
}

void write_scope_start(std::ofstream& out)
{
    out << "{\n";
}

void write_scope_end(std::ofstream& out)
{
    out << "}\n";
}

void write_function_declaration(const std::string& functionNamespace, const std::string& functionName,
    std::ofstream& out)
{
    std::string declaration = "std::string " + functionNamespace + "::" + functionName + "()\n";
    out << declaration;
}

void write_function_body(char* buffer, std::ofstream& out)
{
    out << "return ";
    out << "\"";
    char* c = buffer;
    while (c[0] != '\0')
    {
        if (c[0] == '\n')
        {
            out << "\\n\"\n";
            out << "\"";
        }
        else
        {
            out << c[0];
        }

        c++;
    }
    out << "\\n\";\n";
}

void generate_cpp_file(std::vector<File>& files)
{
    std::ofstream out;
    out.open("../../src/graphics/platform/directx/HLSL/hlsl_shaders.cpp");

    if (!out.is_open())
    {
        return;
    }

    std::unordered_map<std::string, std::string> filePathToFunctionNameMap;
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/geometry_v.hlsl"] = "getGeometryVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/geometry_f.hlsl"] = "getGeometryFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/ssao_v.hlsl"] = "getSSAOVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/ssao_f.hlsl"] = "getSSAOFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/shadow_depth_map_v.hlsl"] = "getShadowDepthMapVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/shadow_depth_map_f.hlsl"] = "getShadowDepthMapFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/shadow_depth_cubemap_v.hlsl"] = "getShadowDepthCubemapVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/shadow_depth_cubemap_f.hlsl"] = "getShadowDepthCubemapFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/shadow_depth_cubemap_g.hlsl"] = "getShadowDepthCubemapGeometryShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/color_v.hlsl"] = "getColorVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/color_f.hlsl"] = "getColorFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/color_instanced_v.hlsl"] = "getColorInstancedVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/color_instanced_f.hlsl"] = "getColorInstancedFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/screen_quad_v.hlsl"] = "getScreenQuadVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/screen_quad_f.hlsl"] = "getScreenQuadFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/sprite_v.hlsl"] = "getSpriteVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/sprite_f.hlsl"] = "getSpriteFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/gbuffer_v.hlsl"] = "getGBufferVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/gbuffer_f.hlsl"] = "getGBufferFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/normal_v.hlsl"] = "getNormalVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/normal_f.hlsl"] = "getNormalFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/normal_instanced_v.hlsl"] = "getNormalInstancedVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/normal_instanced_f.hlsl"] = "getNormalInstancedFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/position_v.hlsl"] = "getPositionVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/position_f.hlsl"] = "getPositionFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/position_instanced_v.hlsl"] = "getPositionInstancedVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/position_instanced_f.hlsl"] = "getPositionInstancedFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/linear_depth_v.hlsl"] = "getLinearDepthVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/linear_depth_f.hlsl"] = "getLinearDepthFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/linear_depth_instanced_v.hlsl"] = "getLinearDepthInstancedVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/linear_depth_instanced_f.hlsl"] = "getLinearDepthInstancedFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/line_v.hlsl"] = "getLineVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/line_f.hlsl"] = "getLineFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/gizmo_v.hlsl"] = "getGizmoVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/gizmo_f.hlsl"] = "getGizmoFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/grid_v.hlsl"] = "getGridVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/grid_f.hlsl"] = "getGridFragmentShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/standard_v.hlsl"] = "getStandardVertexShader";
    filePathToFunctionNameMap["../../src/graphics/platform/directx/hlsl/standard_f.hlsl"] = "getStandardFragmentShader";

    write_header("hlsl", out);

    std::cout << "Generating shader cpp file..." << std::endl;
    for (size_t i = 0; i < files.size(); i++)
    {
        auto it = filePathToFunctionNameMap.find(files[i].path);
        if (it != filePathToFunctionNameMap.end())
        {
            std::string functionName = filePathToFunctionNameMap[files[i].path];

            std::cout << "functionName: " << functionName << " path: " << files[i].path << std::endl;

            write_function_declaration("hlsl", functionName, out);
            write_scope_start(out);
            write_function_body(files[i].buffer, out);
            write_scope_end(out);
        }
    }

    out.close();
}

int main()
{
    std::vector<File> files = loadFilesInDirectory("../../src/graphics/platform/directx/hlsl/");

    generate_cpp_file(files);

    for (size_t i = 0; i < files.size(); i++)
    {
        free(files[i].buffer);
    }

    return 0;
}