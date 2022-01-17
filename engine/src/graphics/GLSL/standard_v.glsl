#version 430 core
#include "camera.glsl"
#include "light.glsl"

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;
#if defined (INSTANCING)
layout (location = 3) in mat4 model;
#else
uniform mat4 model;
#endif

out vec3 FragPos;
out vec3 CameraPos;
out vec3 Normal;
out vec2 TexCoord;
out float ClipSpaceZ;
out vec4 FragPosLightSpace[5];
void main()
{
    CameraPos = Camera.cameraPos;
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;
    TexCoord = texCoord;
    gl_Position = Camera.viewProjection * vec4(FragPos, 1.0);
    ClipSpaceZ = gl_Position.z;
    for (int i = 0; i < 5; i++)
    {
        FragPosLightSpace[i] = Light.lightProjection[i] * Light.lightView[i] * vec4(FragPos, 1.0f);
    }
};