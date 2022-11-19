#version 430 core
#include "camera.glsl"

layout (location = 0) in vec3 position;
uniform mat4 model;
void main()
{
    gl_Position = Camera.viewProjection * model * vec4(position, 1.0);
}