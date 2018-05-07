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
in vec2 atexture;

out vec2 TexCoord;

uniform mat4 model;
//uniform mat4 view;
//uniform mat4 projection;
uniform float pointRadius;
uniform float pointScale; 

void main()
{
    vec3 pos = vec3(view * model * vec4(position.x, position.y, position.z, 1.0f));

    gl_PointSize = pointRadius * (pointScale / length(pos));

	gl_Position = projection * view * model * vec4(position.x, position.y, position.z, 1.0f);

    TexCoord = atexture;
}