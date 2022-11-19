#version 430 core
#include "camera.glsl"
layout (location = 0) in vec3 position;
layout(location = 3) in mat4 model;
void main()
{
    gl_Position = Camera.viewProjection * model * vec4(position, 1.0);
}