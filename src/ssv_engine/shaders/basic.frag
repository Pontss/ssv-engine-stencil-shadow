#version 330 core
in vec3 v_normal;
in vec3 v_world_position;
in vec2 v_uv;
uniform vec3 u_color;
uniform vec3 u_light_position;
uniform vec3 u_camera_position;
uniform sampler2D u_material;
uniform float u_roughness;
uniform float u_emissive;
out vec4 frag_color;
void main() {
    vec3 albedo = texture(u_material, v_uv).rgb * u_color;
    vec3 n = normalize(v_normal);
    vec2 texel = 1.0 / vec2(textureSize(u_material, 0));
    float height_center = texture(u_material, v_uv).a;
    float height_x = texture(u_material, v_uv + vec2(texel.x, 0.0)).a;
    float height_y = texture(u_material, v_uv + vec2(0.0, texel.y)).a;
    vec3 bump = vec3((height_x - height_center) * 2.8, (height_y - height_center) * 2.8, 0.0);
    n = normalize(n - bump);
    vec3 to_light = u_light_position - v_world_position;
    float distance_to_light = length(to_light);
    vec3 l = normalize(to_light);
    vec3 v = normalize(u_camera_position - v_world_position);
    vec3 h = normalize(l + v);
    float diffuse = max(dot(n, l), 0.0);
    float attenuation = 1.0 / (1.0 + 0.025 * distance_to_light + 0.008 * distance_to_light * distance_to_light);
    float sky_fill = 0.42 + 0.18 * max(n.y, 0.0);
    float spec_power = mix(96.0, 12.0, clamp(u_roughness, 0.0, 1.0));
    float specular = pow(max(dot(n, h), 0.0), spec_power) * (1.0 - u_roughness) * 0.82;
    vec3 warm_light = vec3(1.0, 0.86, 0.70);
    vec3 lit = albedo * (sky_fill + diffuse * attenuation * 1.42 * warm_light) + specular * vec3(0.82,0.91,1.0);
    lit = mix(lit, albedo * 2.2, clamp(u_emissive, 0.0, 1.0));
    float fog = smoothstep(12.0, 30.0, length(u_camera_position - v_world_position));
    vec3 fog_color = vec3(0.105, 0.115, 0.125);
    frag_color = vec4(mix(lit, fog_color, fog * 0.72), 1.0);
}
