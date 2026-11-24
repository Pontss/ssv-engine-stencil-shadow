#version 330 core
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_normal;
out vec3 v_world_position;
out vec2 v_uv;
void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_world_position = world.xyz;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
