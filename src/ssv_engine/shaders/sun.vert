#version 330 core
in vec2 in_corner;
uniform mat4 u_view_projection;
uniform vec3 u_center;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform float u_radius;
out vec2 v_uv;
void main() {
    vec3 world = u_center
        + u_camera_right * in_corner.x * u_radius
        + u_camera_up * in_corner.y * u_radius;
    v_uv = in_corner * 0.5 + 0.5;
    gl_Position = u_view_projection * vec4(world, 1.0);
}
