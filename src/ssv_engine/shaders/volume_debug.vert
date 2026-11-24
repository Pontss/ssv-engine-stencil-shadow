#version 330 core
in vec4 in_position;
uniform mat4 u_view_projection;
void main() {
    gl_Position = u_view_projection * in_position;
}
