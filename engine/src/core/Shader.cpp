#include <iostream>

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Shader.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

std::string Shader::lineVertexShader = 
"layout (std140) uniform CameraBlock\n"
"{\n"
"	mat4 projection;\n"
"	mat4 view;\n"
"	vec3 cameraPos;\n"
"}Camera;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = Camera.projection * Camera.view * vec4(position, 1.0);\n"
"}";

std::string Shader::lineFragmentShader = 
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
"}";


std::string Shader::colorVertexShader = 
"layout (std140) uniform CameraBlock\n"
"{\n"
"	mat4 projection;\n"
"	mat4 view;\n"
"	vec3 cameraPos;\n"
"}Camera;\n"
"uniform mat4 model;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);\n"
"}";

std::string Shader::colorFragmentShader = 
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
"}";






std::string Shader::graphVertexShader = 
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(position, 1.0);\n"
"}";

std::string Shader::graphFragmentShader = 
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
"}";

std::string Shader::windowVertexShader = 
"in vec3 position;\n"
"in vec2 texCoord;\n"
"out vec2 TexCoord;\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(position, 1.0);\n"
"   TexCoord = texCoord;\n"
"}";

std::string Shader::windowFragmentShader = 
"uniform sampler2D texture0;\n"
"in vec2 TexCoord;\n"
"out vec4 FragColor;\n" 
"void main()\n"
"{\n"
"    FragColor = texture(texture0, TexCoord);\n"
"}";

std::string Shader::normalMapVertexShader = 
"layout (std140) uniform CameraBlock\n"
"{\n"
"	mat4 projection;\n"
"	mat4 view;\n"
"	vec3 cameraPos;\n"
"}Camera;\n"
"uniform mat4 model;\n"
"in vec3 position;\n"
"in vec3 normal;\n"
"out vec3 Normal;\n"
"void main()\n"
"{\n"
"	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);\n"
"   Normal = normal;\n"
"}";

std::string Shader::normalMapFragmentShader = 
"in vec3 Normal;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(Normal.xyz, 1.0f);\n"
"}";

std::string Shader::depthMapVertexShader = 
"layout (std140) uniform CameraBlock\n"
"{\n"
"	mat4 projection;\n"
"	mat4 view;\n"
"	vec3 cameraPos;\n"
"}Camera;\n"
"uniform mat4 model;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);\n"
"}";

std::string Shader::depthMapFragmentShader = 
"void main()\n"
"{\n"
"}";

std::string Shader::shadowDepthMapVertexShader = 
"uniform mat4 projection;\n"
"uniform mat4 view;\n"
"uniform mat4 model;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = projection * view * model * vec4(position, 1.0);\n"
"}";

std::string Shader::shadowDepthMapFragmentShader = 
"void main()\n"
"{\n"
"}";

std::string Shader::overdrawVertexShader = 
"layout (std140) uniform CameraBlock\n"
"{\n"
"	mat4 projection;\n"
"	mat4 view;\n"
"	vec3 cameraPos;\n"
"}Camera;\n"
"uniform mat4 model;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);\n"
"}";

std::string Shader::overdrawFragmentShader = 
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(1.0, 0.0, 0.0, 0.1);\n"
"}";

std::string Shader::fontVertexShader = 
"layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>\n"
"out vec2 TexCoords;\n"
"uniform mat4 projection;\n"
"void main()\n"
"{\n"
"    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);\n"
"    TexCoords = vertex.zw;\n"
"}";

std::string Shader::fontFragmentShader = 
"in vec2 TexCoords;\n"
"out vec4 color;\n"
"uniform sampler2D text;\n"
"uniform vec3 textColor;\n"
"void main()\n"
"{\n"    
"    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);\n"
"    color = vec4(textColor, 1.0) * sampled;\n"
"}";

std::string Shader::instanceVertexShader = 
"out vec4 FragColor;\n" 
"in vec3 fColor;\n"
"void main()\n"
"{\n"
"    FragColor = vec4(fColor, 1.0);\n"
"}";

std::string Shader::instanceFragmentShader = 
"layout (location = 0) in vec2 aPos;\n"
"layout (location = 1) in vec3 aColor;\n"
"layout (location = 2) in vec2 aOffset;\n"
"out vec3 fColor;\n"
"void main()\n"
"{\n"
"    gl_Position = vec4(aPos + aOffset, 0.0, 1.0);\n"
"    fColor = aColor;\n"
"}";





std::string Shader::gbufferVertexShader = 
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec3 aNormal;\n"
"layout (location = 2) in vec2 aTexCoords;\n"

"out vec3 FragPos;\n"
"out vec2 TexCoords;\n"
"out vec3 Normal;\n"

"uniform mat4 model;\n"
"uniform mat4 view;\n"
"uniform mat4 projection;\n"

"void main()\n"
"{\n"
"    vec4 worldPos = model * vec4(aPos, 1.0);\n"
"    FragPos = worldPos.xyz;\n" 
"    TexCoords = aTexCoords;\n"
    
"    mat3 normalMatrix = transpose(inverse(mat3(model)));\n"
"    Normal = normalMatrix * aNormal;\n"

"    gl_Position = projection * view * worldPos;\n"
"}\n";

std::string Shader::gbufferFragmentShader = 
"layout (location = 0) out vec3 gPosition;\n"
"layout (location = 1) out vec3 gNormal;\n"
"layout (location = 2) out vec4 gAlbedoSpec;\n"

"in vec2 TexCoords;\n"
"in vec3 FragPos;\n"
"in vec3 Normal;\n"

"uniform sampler2D texture_diffuse1;\n"
"uniform sampler2D texture_specular1;\n"

"void main()\n"
"{\n"    
"    // store the fragment position vector in the first gbuffer texture\n"
"    gPosition = FragPos;\n"
"    // also store the per-fragment normals into the gbuffer\n"
"    gNormal = normalize(Normal);\n"
"    // and the diffuse per-fragment color\n"
"    gAlbedoSpec.rgb = texture(texture_diffuse1, TexCoords).rgb;\n"
"    // store specular intensity in gAlbedoSpec's alpha component\n"
"    gAlbedoSpec.a = texture(texture_specular1, TexCoords).r;\n"
"}\n";



Shader::Shader()
{
	programCompiled = false;
	assetId = Guid::INVALID;
}

Shader::Shader(std::vector<char> data)
{
	size_t index = sizeof(int);
	ShaderHeader* header = reinterpret_cast<ShaderHeader*>(&data[index]);

	assetId = header->shaderId;
	size_t vertexShaderSize = header->vertexShaderSize;
	size_t geometryShaderSize = header->geometryShaderSize;
	size_t fragmentShaderSize = header->fragmentShaderSize;

	index += sizeof(ShaderHeader);

	std::vector<char>::iterator start = data.begin();
	std::vector<char>::iterator end = data.begin();
	start += index;
	end += index + vertexShaderSize;

	vertexShader = std::string(start, end);

	start +=vertexShaderSize;
	end += geometryShaderSize;

	geometryShader = std::string(start, end);

	start += geometryShaderSize;
	end += fragmentShaderSize;

	fragmentShader = std::string(start, end);

	index += vertexShaderSize + geometryShaderSize + fragmentShaderSize;

	programCompiled = false;

	std::cout << "shader index: " << index << " data size: " << data.size() << std::endl;
}

Shader::~Shader()
{

}

bool Shader::isCompiled()
{
	return programCompiled;
}

void Shader::compile()
{
	Graphics::compile(this);
}

void Shader::setUniformBlock(std::string blockName, int bindingPoint)
{
	Graphics::setUniformBlock(this, blockName, bindingPoint);
}

void Shader::setBool(std::string name, ShaderVariant variant, bool value)
{
	Graphics::setBool(this, variant, name, value);
}

void Shader::setInt(std::string name, ShaderVariant variant, int value)
{
	Graphics::setInt(this, variant, name, value);
}

void Shader::setFloat(std::string name, ShaderVariant variant, float value)
{
	Graphics::setFloat(this, variant, name, value);
}

void Shader::setVec2(std::string name, ShaderVariant variant, glm::vec2 &vec)
{
	Graphics::setVec2(this, variant, name, vec);
}

void Shader::setVec3(std::string name, ShaderVariant variant, glm::vec3 &vec) 
{
	Graphics::setVec3(this, variant, name, vec);
}

void Shader::setVec4(std::string name, ShaderVariant variant, glm::vec4 &vec)
{
	Graphics::setVec4(this, variant, name, vec);
}

void Shader::setMat2(std::string name, ShaderVariant variant, glm::mat2 &mat)
{
	Graphics::setMat2(this, variant, name, mat);
}

void Shader::setMat3(std::string name, ShaderVariant variant, glm::mat3 &mat)
{
	Graphics::setMat3(this, variant, name, mat);
}

void Shader::setMat4(std::string name, ShaderVariant variant, glm::mat4 &mat)
{
	Graphics::setMat4(this, variant, name, mat);
}
