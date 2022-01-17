#version 430 core
#include "camera.glsl"
layout(location = 3) in mat4 model;

in vec3 position;
void main()
{
    gl_Position = Camera.viewProjection * model * vec4(position, 1.0);
}