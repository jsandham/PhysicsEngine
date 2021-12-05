#version 430 core
layout(std140) uniform CameraBlock
{
    mat4 projection;
    mat4 view;
    vec3 cameraPos;
}Camera;
layout(std140) uniform LightBlock
{
    mat4 lightProjection[5]; // 0    64   128  192  256
    mat4 lightView[5]; // 320  384  448  512  576
    vec3 position; // 640
    vec3 direction; // 656
    vec3 color; // 672
    float cascadeEnds[5]; // 688  704  720  736  752
    float intensity; // 768
    float spotAngle; // 772
    float innerSpotAngle; // 776
    float shadowNearPlane; // 780
    float shadowFarPlane; // 784
    float shadowBias; // 788
    float shadowRadius; // 792
    float shadowStrength; // 796
}Light;
uniform mat4 model;
in vec3 position;
in vec3 normal;
in vec2 texCoord;
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
    gl_Position = Camera.projection * Camera.view * vec4(FragPos, 1.0);
    ClipSpaceZ = gl_Position.z;
    for (int i = 0; i < 5; i++)
    {
        FragPosLightSpace[i] = Light.lightProjection[i] * Light.lightView[i] * vec4(FragPos, 1.0f);
    }
};