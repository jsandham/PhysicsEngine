in vec3 position;
in vec2 texCoord;
out vec2 TexCoord;
void main()
{
   gl_Position = vec4(position, 1.0);
   TexCoord = texCoord;
}
