#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

void write_header(const std::string& name, std::ofstream& out)
{
    out << "//***************************************\n";
    out << "// THIS IS A GENERATED FILE. DO NOT EDIT.\n";
    out << "//***************************************\n";
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
    
    std::cout << "Reading geometry_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGeometryVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/geometry_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading geometry_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGeometryFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/geometry_f.glsl", out);
    write_scope_end(out);

    std::cout << "Reading ssao_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getSSAOVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/ssao_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading ssao_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getSSAOFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/ssao_f.glsl", out);
    write_scope_end(out);

    std::cout << "Reading shadow_depth_map_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthMapVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/shadow_depth_map_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading shadow_depth_map_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthMapFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/shadow_depth_map_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading shadow_depth_cubemap_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthCubemapVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/shadow_depth_cubemap_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading shadow_depth_cubemap_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthCubemapFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/shadow_depth_cubemap_f.glsl", out);
    write_scope_end(out);

    std::cout << "Reading shadow_depth_cubemap_g.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthCubemapGeometryShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/shadow_depth_cubemap_g.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading color_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getColorVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/color_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading color_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getColorFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/color_f.glsl", out);
    write_scope_end(out);

    std::cout << "Reading screen_quad_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getScreenQuadVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/screen_quad_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading screen_quad_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getScreenQuadFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/screen_quad_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading sprite_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getSpriteVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/sprite_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading sprite_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getSpriteFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/sprite_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading gbuffer_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGBufferVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/gbuffer_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading gbuffer_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGBufferFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/gbuffer_f.glsl", out);
    write_scope_end(out);
   
    std::cout << "Reading normal_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getNormalVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/normal_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading normal_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getNormalFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/normal_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading position_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getPositionVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/position_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading position_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getPositionFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/position_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading linear_depth_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getLinearDepthVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/linear_depth_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading linear_depth_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getLinearDepthFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/linear_depth_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading line_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getLineVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/line_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading line_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getLineFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/line_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading gizmo_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGizmoVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/gizmo_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading gizmo_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGizmoFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/gizmo_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading grid_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGridVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/grid_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading grid_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGridFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/grid_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading standard_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getStandardVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/standard_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading standard_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getStandardFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/standard_f.glsl", out);
    write_scope_end(out);

    out.close();
}


int main()
{
    generate_shader_cpp_file();
	return 0;
}