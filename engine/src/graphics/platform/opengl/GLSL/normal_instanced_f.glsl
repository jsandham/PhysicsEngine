#version 430 core
in vec3 Normal;
out vec4 FragColor;
void main()
{
  FragColor = vec4(normalize(Normal), 1);
}