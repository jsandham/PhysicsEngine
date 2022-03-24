#version 430 core
#include "camera.glsl"

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;
out vec3 FragPos;
out vec3 Normal;
uniform mat4 model;
void main()
{
    vec4 viewPos = Camera.view * model * vec4(position, 1.0);
    FragPos = viewPos.xyz;
    mat3 normalMatrix = transpose(inverse(mat3(Camera.view * model)));
    Normal = normalMatrix * normal;
    gl_Position = Camera.projection * viewPos;
}