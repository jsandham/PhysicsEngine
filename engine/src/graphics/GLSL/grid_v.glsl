#version 430 core
#include "camera.glsl"

uniform mat4 mvp;
uniform vec4 color;
in vec3 position;
out vec4 Color;
void main()
{
    gl_Position = mvp * vec4(position, 1.0);
    Color = color;
}