VERTEX:

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;
layout (location = 3) in mat4 instanceModel;

uniform mat4 projection;
uniform mat4 view;

//out vec2 TexCoord;

void main()
{
    gl_Position = projection * view * instanceModel * vec4(position, 1.0); 

    //TexCoords = texCoord;
}


FRAGMENT:

//in vec2 TexCoord;

out vec4 FragColor;

void main(void) 
{
	FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}