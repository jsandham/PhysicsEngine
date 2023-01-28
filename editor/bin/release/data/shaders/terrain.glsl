#vertex
#version 430 core
in vec3 position;
out float height;
void main()
{
    height = position.y;
    gl_Position = vec4(position, 1.0);
}

#fragment
#version 430 core
out vec4 FragColor;
in float height;
void main()
{
    FragColor = vec4(height, height, height, 1);
}