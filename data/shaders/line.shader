VERTEX:

#version 330 core

layout (std140) uniform CameraBlock
{
	mat4 projection;
	mat4 view;
	vec3 cameraPos;
};

layout (std140) uniform LightBlock
{
	mat4 lightProjection[5];
	mat4 lightView[5];
};

in vec3 position;

uniform mat4 model;
//uniform mat4 view;
//uniform mat4 projection;

void main()
{
	//gl_Position = projection * view * model * vec4(position, 1.0);
	gl_Position = projection * view * vec4(position, 1.0);
}


FRAGMENT:

#version 330 core

out vec4 FragColor;

void main()
{
	FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}