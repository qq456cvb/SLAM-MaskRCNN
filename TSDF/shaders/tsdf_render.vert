#version 400
in vec2 aPosCoord;
in vec2 aTexCoord;

out vec2 vTexCoord;

void main(void)
{
    gl_Position = vec4(aPosCoord, 0, 1.0);
    vTexCoord = aTexCoord;
}