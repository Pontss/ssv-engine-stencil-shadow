#version 330 core
uniform float u_viewport_height;
out vec4 frag_color;
void main() {
    float horizon = gl_FragCoord.y / max(u_viewport_height, 1.0);
    vec3 low = vec3(0.24, 0.19, 0.15);
    vec3 high = vec3(0.035, 0.055, 0.095);
    vec3 color = mix(low, high, clamp(horizon, 0.0, 1.0));
    float glow = exp(-pow((horizon - 0.30) * 8.0, 2.0));
    color += vec3(0.42, 0.19, 0.065) * glow * 0.62;
    float sun_glow = exp(-pow((horizon - 0.34) * 13.0, 2.0));
    color += vec3(0.24, 0.12, 0.055) * sun_glow;
    frag_color = vec4(color, 1.0);
}
