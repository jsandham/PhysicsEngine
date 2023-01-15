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
layout (location = 2) in vec2 texCoord;
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
void main()
{
	FragPos = vec3(model * vec4(position, 1.0));
	Normal = normalize(normal);
	TexCoord = texCoord;
	gl_Position = Camera.projection * Camera.view * vec4(FragPos, 1.0);
}

#fragment
#version 430 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
out vec4 FragColor;
void main(void)
{
	FragColor = vec4(Normal, 1.0f);
}
