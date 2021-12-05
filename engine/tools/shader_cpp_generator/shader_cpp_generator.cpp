#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

void write_header(const std::string& name, std::ofstream& out)
{
    out << "#include <string>\n";
    out << "#include \"glsl_shaders.h\"\n";
    out << ("using namespace " + name + ";\n");
}

void write_scope_start(std::ofstream &out)
{
    out << "{\n";
}

void write_scope_end(std::ofstream &out)
{
    out << "}\n";
}

void write_function_declaration(const std::string& functionNamespace, const std::string& functionName, std::ofstream& out)
{
    std::string declaration = "std::string " + functionNamespace + "::" + functionName + "()\n";
    out << declaration;
}

void write_function_body(const std::string& shaderFilepath, std::ofstream& out)
{
    std::ifstream in;
    in.open(shaderFilepath, std::ifstream::in);

    if (!in.is_open())
    {
        return;
    }

    std::stringstream buffer;
    buffer << in.rdbuf();
    in.close();

    out << "return ";

    std::string line;
    while (std::getline(buffer, line))
    {
        out << ("\"" + line + "\\n\"\n");
    }
    out << ";\n";

    //out << buffer.str();
}

std::string generate_shader_function(const std::string &functionName, const std::string &shaderFilepath)
{
    return "";
}

void generate_shader_cpp_file()
{
    std::ofstream out;
    out.open("../../src/graphics/GLSL/glsl_shaders.cpp");

    if (!out.is_open())
    {
        return;
    }

    write_header("PhysicsEngine", out);
  
    std::cout << "Generating shader cpp file..." << std::endl;
    
    std::cout << "Reading geometry.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getGeometryVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/geometry.vs", out);
    write_scope_end(out);

    std::cout << "Reading geometry.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getGeometryFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/geometry.fs", out);
    write_scope_end(out);

    std::cout << "Reading ssao.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getSSAOVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/ssao.vs", out);
    write_scope_end(out);

    std::cout << "Reading ssao.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getSSAOFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/ssao.fs", out);
    write_scope_end(out);

    std::cout << "Reading shadow_depth_map.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthMapVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/shadow_depth_map.vs", out);
    write_scope_end(out);

    std::cout << "Reading shadow_depth_map.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthMapFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/shadow_depth_map.fs", out);
    write_scope_end(out);
    
    std::cout << "Reading shadow_depth_cubemap.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthCubemapVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/shadow_depth_cubemap.vs", out);
    write_scope_end(out);

    std::cout << "Reading shadow_depth_cubemap.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthCubemapFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/shadow_depth_cubemap.fs", out);
    write_scope_end(out);

    std::cout << "Reading shadow_depth_cubemap.gs" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthCubemapGeometryShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/shadow_depth_cubemap.gs", out);
    write_scope_end(out);
    
    std::cout << "Reading color.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getColorVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/color.vs", out);
    write_scope_end(out);

    std::cout << "Reading color.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getColorFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/color.fs", out);
    write_scope_end(out);

    std::cout << "Reading screen_quad.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getScreenQuadVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/screen_quad.vs", out);
    write_scope_end(out);

    std::cout << "Reading screen_quad.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getScreenQuadFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/screen_quad.fs", out);
    write_scope_end(out);
    
    std::cout << "Reading sprite.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getSpriteVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/sprite.vs", out);
    write_scope_end(out);

    std::cout << "Reading sprite.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getSpriteFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/sprite.fs", out);
    write_scope_end(out);
    
    std::cout << "Reading gbuffer.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getGBufferVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/gbuffer.vs", out);
    write_scope_end(out);

    std::cout << "Reading gbuffer.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getGBufferFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/gbuffer.fs", out);
    write_scope_end(out);
   
    std::cout << "Reading normal.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getNormalVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/normal.vs", out);
    write_scope_end(out);

    std::cout << "Reading normal.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getNormalFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/normal.fs", out);
    write_scope_end(out);
    
    std::cout << "Reading position.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getPositionVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/position.vs", out);
    write_scope_end(out);

    std::cout << "Reading position.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getPositionFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/position.fs", out);
    write_scope_end(out);
    
    std::cout << "Reading linear_depth.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getLinearDepthVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/linear_depth.vs", out);
    write_scope_end(out);

    std::cout << "Reading linear_depth.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getLinearDepthFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/linear_depth.fs", out);
    write_scope_end(out);
    
    std::cout << "Reading line.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getLineVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/line.vs", out);
    write_scope_end(out);

    std::cout << "Reading line.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getLineFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/line.fs", out);
    write_scope_end(out);
    
    std::cout << "Reading gizmo.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getGizmoVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/gizmo.vs", out);
    write_scope_end(out);

    std::cout << "Reading gizmo.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getGizmoFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/gizmo.fs", out);
    write_scope_end(out);
    
    std::cout << "Reading grid.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getGridVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/grid.vs", out);
    write_scope_end(out);

    std::cout << "Reading grid.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getGridFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/grid.fs", out);
    write_scope_end(out);
    
    std::cout << "Reading standard.vs" << std::endl;
    write_function_declaration("PhysicsEngine", "getStandardVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/standard.vs", out);
    write_scope_end(out);

    std::cout << "Reading standard.fs" << std::endl;
    write_function_declaration("PhysicsEngine", "getStandardFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/GLSL/standard.fs", out);
    write_scope_end(out);

    out.close();
}


int main()
{
    generate_shader_cpp_file();
	return 0;
}