#version 430 core
#include "camera.glsl"

uniform mat4 model;
in vec3 position;
void main()
{
    gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);
}