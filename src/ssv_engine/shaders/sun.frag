#version 330 core
in vec2 v_uv;
out vec4 frag_color;
void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    float r = length(p);
    float disc = 1.0 - smoothstep(0.78, 0.94, r);
    float corona = exp(-max(r - 0.58, 0.0) * 8.0) * (1.0 - smoothstep(0.88, 1.0, r));
    float core = 1.0 - smoothstep(0.0, 0.72, r);
    vec3 color = mix(vec3(1.0, 0.30, 0.035), vec3(1.0, 0.88, 0.48), core);
    color = mix(color, vec3(1.0, 0.98, 0.78), pow(core, 5.0));
    float alpha = max(disc, corona * 0.42);
    if (alpha < 0.01) discard;
    frag_color = vec4(color * (1.0 + corona * 0.65), alpha);
}
