#version 430 core
#include "occlusion.glsl"
layout(location = 0) in vec3 aPos;
layout(location = 1) in int modelIndex;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    /*gl_Position = Camera.viewProjection * vec4(aPos, 1.0f);*/
    gl_Position = projection * view * Occlusion.models[modelIndex] * vec4(aPos, 1.0f);
}