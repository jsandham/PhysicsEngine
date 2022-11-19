#version 430 core
#include "camera.glsl"

layout (location = 0) in vec3 position;

uniform mat4 mvp;
uniform vec4 color;
out vec4 Color;
void main()
{
    gl_Position = mvp * vec4(position, 1.0);
    Color = color;
}