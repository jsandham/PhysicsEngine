#vertex
#version 430 core
layout(std140) uniform CameraBlock
{
    mat4 projection;
    mat4 view;
    mat4 viewProjection;
    vec3 cameraPos;
}Camera;
uniform mat4 model;
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
out vec3 Normal;
void main()
{
	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);
   Normal = normal;
}

#fragment
#version 430 core
in vec3 Normal;
out vec4 FragColor;
void main()
{
	FragColor = vec4(Normal.xyz, 1.0f);
}
