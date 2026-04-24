#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "stb_image.h"

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679f

const char* vertex_src =
    "#version 330 core\n"

    "layout(location = 0) in vec3 a_position;\n"
    "layout(location = 1) in vec4 a_color;\n"
    "layout(location = 2) in vec2 a_tex_coord;\n"
    "layout(location = 3) in vec3 a_normal;\n"
    "layout(location = 4) in float a_tex_index;\n"

    "uniform mat4 u_model;\n"
    "uniform mat4 u_view;\n"
    "uniform mat4 u_projection;\n"
    
    "out vec4 v_color;\n"
    "out vec2 v_tex_coord;\n"
    "out vec3 v_normal;\n"
    "out vec3 v_frag_pos;\n"
    "out float v_tex_index;\n"
    
    "void main()\n"
    "{\n"
    "    vec4 world_pos = u_model * vec4(a_position, 1.0);\n"
    "    gl_Position = u_projection * u_view * world_pos;\n"

    "    v_frag_pos = world_pos.xyz;\n"

    "    v_normal = mat3(transpose(inverse(u_model))) * a_normal;\n"

    "    v_color     = a_color;\n"
    "    v_tex_coord = a_tex_coord;\n"
    "    v_tex_index = a_tex_index;\n"
    "}\n";

const char* fragment_src =
    "#version 330 core\n"

    "in vec4  v_color;\n"
    "in vec2  v_tex_coord;\n"
    "in vec3  v_normal;\n"
    "in vec3  v_frag_pos;\n"

    "uniform vec3 u_light_pos;\n"
    "uniform vec3 u_light_color;\n"
    "uniform sampler2D u_texture;\n"
    "uniform int u_use_texture;\n"

    "out vec4 frag_color;\n"

    "void main()\n"
    "{\n"
    "    float ambient_strength = 0.1;\n"
    "    vec3  ambient = ambient_strength * u_light_color;\n"

    "    vec3 norm = normalize(v_normal);\n"
    "    vec3 light_dir = normalize(u_light_pos - v_frag_pos);\n"
    "    float diff = max(dot(norm, light_dir), 0.0);\n"
    "    vec3 diffuse = diff * u_light_color;\n"

    "    vec4 base_color;\n"
    "    if (u_use_texture == 1)\n"
    "        base_color = texture(u_texture, v_tex_coord);\n"
    "    else\n"
    "        base_color = v_color;\n"

    "    vec3 final_color = base_color.rgb * (diffuse + ambient);\n"
    "    frag_color = vec4(final_color, base_color.a);\n"
    "}\n";


// ------------------------------------------------------------------
// tipos
// ------------------------------------------------------------------

typedef struct { float x, y, z;       } vec3;
typedef struct { float x, y, z, w;    } vec4;
typedef struct { float x, y;          } vec2;
typedef struct { float m[16];         } mat4;

// ------------------------------------------------------------------
// declarações
// ------------------------------------------------------------------

const vec3 MAIN_LIGHT_POS = (vec3){ 10.0, 10.0, 0.0 };
const vec3 MAIN_LIGHT_COLOR = (vec3){ 1.0, 1.0, 1.0 };

vec3  vec3_zero();
vec3  vec3_right();
vec3  vec3_up();
vec3  vec3_forward();
vec3  vec3_one();
vec3  vec3_add(vec3* a, vec3* b);
vec3  vec3_subtract(vec3* a, vec3* b);
vec3  vec3_cross(vec3* a, vec3* b);
float vec3_dot(vec3* a, vec3* b);

mat4 mat4_identity();
mat4 mat4_multiply(mat4* a, mat4* b);
mat4 mat4_translate(mat4* mat, vec3* vec);
mat4 mat4_rotate(mat4* mat, vec3* axis, float angle);
mat4 mat4_rotate_x(mat4* mat, float angle);
mat4 mat4_rotate_y(mat4* mat, float angle);
mat4 mat4_rotate_z(mat4* mat, float angle);
mat4 mat4_scale(mat4* mat, vec3* vec);
void mat4_print(mat4* mat);
mat4 mat4_perspective(float fov, float aspect,
                              float near, float far);

mat4 mat4_look_at(vec3 eye, vec3 center, vec3 up);

// ------------------------------------------------------------------
// math interno (renderer)
// ------------------------------------------------------------------

static void mat4_identity_raw(float out[16])
{
    memset(out, 0, 16 * sizeof(float));
    out[0] = out[5] = out[10] = out[15] = 1.0f;
}

// FIX: corrigido para column-major (j*4+i ao invés de i*4+j)
static void mat4_mul_raw(float out[16], float a[16], float b[16])
{
    float tmp[16] = {0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                tmp[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
    memcpy(out, tmp, 16 * sizeof(float));
}

static void build_translate(float out[16], vec3 pos)
{
    mat4_identity_raw(out);
    out[12] = pos.x;
    out[13] = pos.y;
    out[14] = pos.z;
}

static void build_scale(float out[16], float sx, float sy, float sz)
{
    mat4_identity_raw(out);
    out[0]  = sx;
    out[5]  = sy;
    out[10] = sz;
}

static void build_rotate_x(float out[16], float a)
{
    mat4_identity_raw(out);
    out[5]  =  cosf(a);
    out[9]  = -sinf(a);
    out[6]  =  sinf(a);
    out[10] =  cosf(a);
}

static void build_rotate_y(float out[16], float a)
{
    mat4_identity_raw(out);
    out[0]  =  cosf(a);
    out[8]  =  sinf(a);
    out[2]  = -sinf(a);
    out[10] =  cosf(a);
}

static void build_rotate_z(float out[16], float a)
{
    mat4_identity_raw(out);
    out[0]  =  cosf(a);
    out[4]  = -sinf(a);
    out[1]  =  sinf(a);
    out[5]  =  cosf(a);
}

static void build_transform(float out[16], vec3 pos, vec3 rot, float sx, float sy)
{
    float t[16], rx[16], ry[16], rz[16], s[16], tmp[16];
    build_translate(t,  pos);
    build_rotate_x (rx, rot.x);
    build_rotate_y (ry, rot.y);
    build_rotate_z (rz, rot.z);
    build_scale    (s,  sx, sy, 1.0f);

    mat4_mul_raw(tmp, rz, ry);
    mat4_mul_raw(tmp, tmp, rx);
    mat4_mul_raw(tmp, tmp, s);
    mat4_mul_raw(out, t,   tmp);
}

static vec3 transform_vec4(float m[16], vec4 v)
{
    return (vec3)
    {
        m[0]*v.x + m[4]*v.y + m[8] *v.z + m[12]*v.w,
        m[1]*v.x + m[5]*v.y + m[9] *v.z + m[13]*v.w,
        m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]*v.w,
    };
}

// ------------------------------------------------------------------
// camera helpers
// ------------------------------------------------------------------



// ------------------------------------------------------------------
// input
// ------------------------------------------------------------------

#define MAX_KEYS 512

static GLFWwindow* m_window       = NULL;
static int         m_current_keys[MAX_KEYS];
static int         m_previous_keys[MAX_KEYS];

void input_init(GLFWwindow* window)
{
    m_window = window;
    memset(m_current_keys,  0, sizeof(m_current_keys));
    memset(m_previous_keys, 0, sizeof(m_previous_keys));
}

void input_update()
{
    memcpy(m_previous_keys, m_current_keys, sizeof(m_current_keys));
    for (int i = 0; i < MAX_KEYS; i++)
        m_current_keys[i] = glfwGetKey(m_window, i) == GLFW_PRESS;
}

int input_get_key(int key)
{
    int pressed = glfwGetKey(m_window, key) == GLFW_PRESS;
    m_current_keys[key] = pressed;
    return pressed;
}

int input_get_key_down(int key) { return  m_current_keys[key] && !m_previous_keys[key]; }
int input_get_key_up(int key)   { return !m_current_keys[key] &&  m_previous_keys[key]; }

int input_get_mouse_button(int button)
{
    return glfwGetMouseButton(m_window, button) == GLFW_PRESS;
}

void input_get_mouse_position(double* x, double* y)
{
    glfwGetCursorPos(m_window, x, y);
}

// ------------------------------------------------------------------
// time
// ------------------------------------------------------------------

static double last_time = 0.0;
static float  delta     = 0.0f;

void time_init()
{
    last_time = glfwGetTime();
    delta     = 0.0f;
}

void time_update()
{
    double now = glfwGetTime();
    delta      = (float)(now - last_time);
    last_time  = now;
}

float time_total()
{
    return (float)glfwGetTime();
}

float time_delta()
{
    return delta;
}

// ------------------------------------------------------------------
// .obj parser
// ------------------------------------------------------------------

typedef struct
{
    vec3* positions;   // lidos dos 'v'
    vec2* texcoords;   // lidos dos 'vt'
    vec3* normals;     // lidos dos 'vn'

    int position_count;
    int texcoord_count;
    int normal_count;

    int* pos_indices;
    int* tex_indices;
    int* nor_indices;
    int  face_count;   // número de triângulos
} obj_data_t;

void load_obj(const char* path, obj_data_t* obj)
{
    FILE* fptr = fopen(path, "r");
    if (!fptr)
    {
        printf("Error: could not open %s\n", path);
        return;
    }

    // alocações iniciais generosas
    int cap_v = 65536, cap_vt = 65536, cap_vn = 65536, cap_f = 131072;

    obj->positions  = malloc(cap_v  * sizeof(vec3));
    obj->texcoords  = malloc(cap_vt * sizeof(vec2));
    obj->normals    = malloc(cap_vn * sizeof(vec3));
    obj->pos_indices = malloc(cap_f * 3 * sizeof(int));
    obj->tex_indices = malloc(cap_f * 3 * sizeof(int));
    obj->nor_indices = malloc(cap_f * 3 * sizeof(int));

    obj->position_count = 0;
    obj->texcoord_count = 0;
    obj->normal_count   = 0;
    obj->face_count     = 0;

    char buffer[512];

    while (fgets(buffer, sizeof(buffer), fptr))
    {
        if (strncmp(buffer, "vn ", 3) == 0)
        {
            vec3 n;
            if (sscanf(buffer + 3, "%f %f %f", &n.x, &n.y, &n.z) == 3)
                obj->normals[obj->normal_count++] = n;
        }
        else if (strncmp(buffer, "vt ", 3) == 0)
        {
            vec2 t;
            if (sscanf(buffer + 3, "%f %f", &t.x, &t.y) == 2)
                obj->texcoords[obj->texcoord_count++] = t;
        }
        else if (strncmp(buffer, "v ", 2) == 0)
        {
            vec3 v;
            if (sscanf(buffer + 2, "%f %f %f", &v.x, &v.y, &v.z) == 3)
                obj->positions[obj->position_count++] = v;
        }
        else if (strncmp(buffer, "f ", 2) == 0)
        {
            int pi[3], ti[3], ni[3];

            // tenta "f v/vt/vn v/vt/vn v/vt/vn"
            int r = sscanf(buffer + 2,
                "%d/%d/%d %d/%d/%d %d/%d/%d",
                &pi[0], &ti[0], &ni[0],
                &pi[1], &ti[1], &ni[1],
                &pi[2], &ti[2], &ni[2]);

            if (r != 9)
            {
                // tenta "f v//vn v//vn v//vn" (sem texcoord)
                r = sscanf(buffer + 2,
                    "%d//%d %d//%d %d//%d",
                    &pi[0], &ni[0],
                    &pi[1], &ni[1],
                    &pi[2], &ni[2]);

                ti[0] = ti[1] = ti[2] = 1;
                if (r != 6) continue;
            }

            int base = obj->face_count * 3;
            for (int i = 0; i < 3; i++)
            {
                obj->pos_indices[base + i] = pi[i] - 1;
                obj->tex_indices[base + i] = ti[i] - 1;
                obj->nor_indices[base + i] = ni[i] - 1;
            }
            obj->face_count++;
        }
    }

    fclose(fptr);
    printf("load_obj: %d verts, %d texcoords, %d normals, %d faces\n",
           obj->position_count, obj->texcoord_count,
           obj->normal_count,   obj->face_count);
}

void obj_data_free(obj_data_t* obj)
{
    free(obj->positions);
    free(obj->texcoords);
    free(obj->normals);
    free(obj->pos_indices);
    free(obj->tex_indices);
    free(obj->nor_indices);
    memset(obj, 0, sizeof(obj_data_t));
}

// ------------------------------------------------------------------
// shader
// ------------------------------------------------------------------

typedef struct
{
    unsigned int id;
} shader_t;

static GLint get_uniform_location(GLuint program, const char* name)
{
    GLint location = glGetUniformLocation(program, name);
    if (location == -1)
        printf("Warning: uniform '%s' not found.\n", name);
    return location;
}

static GLint uniform_loc(GLuint prog, const char* name)
{
    GLint loc = glGetUniformLocation(prog, name);
    if (loc == -1)
        printf("Warning: uniform '%s' not found.\n", name);
    return loc;
}

static void set_uniform_mat4(GLuint prog, const char* name, mat4* mat)
{
    glUniformMatrix4fv(uniform_loc(prog, name), 1, GL_FALSE, mat->m);
}

static void set_uniform_int_array(GLuint prog, const char* name,
                                  int* values, int count)
{
    glUniform1iv(uniform_loc(prog, name), count, values);
}
static void set_uniform_float_array(GLuint prog, const char* name,
                                  float* values, int count)
{
    glUniform1fv(uniform_loc(prog, name), count, values);
}
static void set_uniform_vec3(GLuint prog, const char* name, vec3 v)
{
    glUniform3f(uniform_loc(prog, name), v.x, v.y, v.z);
}

static char* read_file(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f)
    {
        printf("ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    char* buf = (char*)malloc(size + 1);
    fread(buf, 1, size, f);
    buf[size] = '\0';
    fclose(f);
    return buf;
}

static shader_t compile_shader(const char* vs, const char* fs)
{
    shader_t shader = {0};
    char info_log[512];
    int  success;

    unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vs, NULL);
    glCompileShader(vertex);
    glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertex, 512, NULL, info_log);
        printf("ERROR::VERTEX\n%s\n", info_log);
    }

    unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fs, NULL);
    glCompileShader(fragment);
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragment, 512, NULL, info_log);
        printf("ERROR::FRAGMENT\n%s\n", info_log);
    }

    shader.id = glCreateProgram();
    glAttachShader(shader.id, vertex);
    glAttachShader(shader.id, fragment);
    glLinkProgram(shader.id);
    glGetProgramiv(shader.id, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shader.id, 512, NULL, info_log);
        printf("ERROR::LINKING\n%s\n", info_log);
    }

    glDeleteShader(vertex);
    glDeleteShader(fragment);
    return shader;
}

shader_t shader_create(const char* vertex_path, const char* fragment_path)
{
    char* vs = read_file(vertex_path);
    char* fs = read_file(fragment_path);
    if (!vs || !fs)
    {
        free(vs);
        free(fs);
        return (shader_t){0};
    }
    shader_t s = compile_shader(vs, fs);
    free(vs); free(fs);
    return s;
}

shader_t shader_create_from_src(const char* vs, const char* fs)
{
    return compile_shader(vs, fs);
}

void shader_use(shader_t* shader)
{
    glUseProgram(shader->id);
}
void shader_set_bool(shader_t* s, const char* name, int value)
{
    glUniform1i(get_uniform_location(s->id, name), value);
}
void shader_set_int(shader_t* s, const char* name, int value)
{
    glUniform1i(get_uniform_location(s->id, name), value);
}
void shader_set_float(shader_t* s, const char* name, float value)
{
    glUniform1f(get_uniform_location(s->id, name), value);
}
void shader_set_int_array(shader_t* s, const char* name, int* values, unsigned int count)
{
    glUniform1iv(get_uniform_location(s->id, name), count, values);
}
void shader_set_float_array(shader_t* s, const char* name, float* values, unsigned int count)
{
    glUniform1fv(get_uniform_location(s->id, name), count, values);
}
void shader_set_mat4(shader_t* s, const char* name, mat4* mat)
{
    glUniformMatrix4fv(get_uniform_location(s->id, name), 1, GL_FALSE, mat->m);
}

// ------------------------------------------------------------------
// vertex / mesh
// ------------------------------------------------------------------

typedef struct
{
    vec3  position;
    vec4  color;
    vec2  tex_coord;
    vec3  normal;
    float tex_index;
} vertex3d_t;

typedef struct
{
    GLuint   vao, vbo, ibo;
    uint32_t index_count;
} mesh_t;

static void setup_vertex_attribs()
{
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex3d_t), (const void*)offsetof(vertex3d_t, position));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(vertex3d_t), (const void*)offsetof(vertex3d_t, color));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex3d_t), (const void*)offsetof(vertex3d_t, tex_coord));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(vertex3d_t), (const void*)offsetof(vertex3d_t, normal));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(vertex3d_t), (const void*)offsetof(vertex3d_t, tex_index));
    glEnableVertexAttribArray(4);
}

mesh_t mesh_create(vertex3d_t* vertices, uint32_t vertex_count,
                   uint32_t*   indices,  uint32_t index_count)
{
    mesh_t mesh = {0};
    mesh.index_count = index_count;

    glGenVertexArrays(1, &mesh.vao);
    glBindVertexArray(mesh.vao);

    glGenBuffers(1, &mesh.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, vertex_count * sizeof(vertex3d_t), vertices, GL_STATIC_DRAW);
    setup_vertex_attribs();

    glGenBuffers(1, &mesh.ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_count * sizeof(uint32_t), indices, GL_STATIC_DRAW);

    glBindVertexArray(0);
    return mesh;
}

void mesh_destroy(mesh_t* mesh)
{
    glDeleteBuffers(1, &mesh->vbo);
    glDeleteBuffers(1, &mesh->ibo);
    glDeleteVertexArrays(1, &mesh->vao);
}

void mesh_draw(mesh_t* mesh)
{
    glBindVertexArray(mesh->vao);
    glDrawElements(GL_TRIANGLES, mesh->index_count, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
}

// ------------------------------------------------------------------
// textures
// ------------------------------------------------------------------

typedef struct
{
    GLuint id;
    int width, height;
} texture_t;

texture_t texture_load(const char* path)
{
    texture_t tex = {0};

    stbi_set_flip_vertically_on_load(1);

    int channels;
    unsigned char* data = stbi_load(path, &tex.width, &tex.height, &channels, 4);

    if (!data)
    {
        printf("Erro ao carregar textura: %s\n", path);
        return tex;
    }

    glGenTextures(1, &tex.id);
    glBindTexture(GL_TEXTURE_2D, tex.id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.width, tex.height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 0);

    printf("Textura carregada: %s (%dx%d)\n", path, tex.width, tex.height);
    return tex;
}

void texture_destroy(texture_t* tex)
{
    glDeleteTextures(1, &tex->id);
    tex->id = 0;
}

void texture_bind(texture_t* tex, int slot)
{
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, tex->id);
}


// ------------------------------------------------------------------
// renderer3d
// ------------------------------------------------------------------

#define MAX_QUADS    1000
#define MAX_VERTICES (MAX_QUADS * 4)
#define MAX_INDICES  (MAX_QUADS * 6)
#define MAX_TEXTURES 16

typedef struct
{
    GLuint vao, vbo, ibo;

    uint32_t    index_count;
    vertex3d_t* vertex_buffer_base;
    vertex3d_t* vertex_buffer_ptr;

    GLuint   texture_slots[MAX_TEXTURES];
    uint32_t texture_slot_index;

    vec4 quad_vertex_positions[4];

    // FIX: cache para reuso no overflow do batch
    GLuint current_shader;
    mat4   current_view;
    mat4   current_projection;
} renderer3d_t;

void renderer3d_init(renderer3d_t* r)
{
    memset(r, 0, sizeof(renderer3d_t));
    r->texture_slot_index = 1;

    r->quad_vertex_positions[0] = (vec4){ -0.5f, -0.5f, 0.0f, 1.0f };
    r->quad_vertex_positions[1] = (vec4){ +0.5f, -0.5f, 0.0f, 1.0f };
    r->quad_vertex_positions[2] = (vec4){ +0.5f, +0.5f, 0.0f, 1.0f };
    r->quad_vertex_positions[3] = (vec4){ -0.5f, +0.5f, 0.0f, 1.0f };

    glGenVertexArrays(1, &r->vao);
    glBindVertexArray(r->vao);

    glGenBuffers(1, &r->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, r->vbo);
    glBufferData(GL_ARRAY_BUFFER, MAX_VERTICES * sizeof(vertex3d_t), NULL, GL_DYNAMIC_DRAW);
    setup_vertex_attribs();

    uint32_t* indices = malloc(MAX_INDICES * sizeof(uint32_t));
    uint32_t  offset  = 0;
    for (uint32_t i = 0; i < MAX_INDICES; i += 6)
    {
        indices[i+0] = offset + 0;
        indices[i+1] = offset + 1;
        indices[i+2] = offset + 2;
        indices[i+3] = offset + 2;
        indices[i+4] = offset + 3;
        indices[i+5] = offset + 0;
        offset += 4;
    }

    glGenBuffers(1, &r->ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, r->ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, MAX_INDICES * sizeof(uint32_t), indices, GL_STATIC_DRAW);
    free(indices);

    r->vertex_buffer_base = malloc(MAX_VERTICES * sizeof(vertex3d_t));
    glBindVertexArray(0);
}

void renderer3d_destroy(renderer3d_t* r)
{
    free(r->vertex_buffer_base);
    glDeleteBuffers(1, &r->vbo);
    glDeleteBuffers(1, &r->ibo);
    glDeleteVertexArrays(1, &r->vao);
}

void renderer3d_flush(renderer3d_t* r)
{
    for (uint32_t i = 0; i < r->texture_slot_index; i++)
    {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, r->texture_slots[i]);
    }
    glBindVertexArray(r->vao);
    glDrawElements(GL_TRIANGLES, r->index_count, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
}

void renderer3d_begin_batch(renderer3d_t* r, GLuint shader_id,
                             mat4* view, mat4* projection)
{
    r->index_count        = 0;
    r->vertex_buffer_ptr  = r->vertex_buffer_base;
    r->texture_slot_index = 1;

    // FIX: cacheia para uso no overflow
    r->current_shader     = shader_id;
    r->current_view       = *view;
    r->current_projection = *projection;

    glUseProgram(shader_id);
    set_uniform_mat4(shader_id, "u_view",       view);
    set_uniform_mat4(shader_id, "u_projection", projection);

    

    // FIX: seta u_model como identidade — transform já está nos vértices
    mat4 identity = mat4_identity();
    set_uniform_mat4(shader_id, "u_model", &identity);
}

void renderer3d_end_batch(renderer3d_t* r)
{
    ptrdiff_t size =
        (uint8_t*)r->vertex_buffer_ptr -
        (uint8_t*)r->vertex_buffer_base;

    glBindBuffer(GL_ARRAY_BUFFER, r->vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size, r->vertex_buffer_base);
    renderer3d_flush(r);
}

void renderer3d_draw_quad(renderer3d_t* r,
                          vec3 position,
                          vec3 rotation,
                          vec2 size,
                          vec3 normal,
                          GLuint texture_id,
                          vec4 color)
{
    // FIX: usa cache ao invés de passar NULL no overflow
    if (r->index_count >= MAX_INDICES)
    {
        renderer3d_end_batch(r);
        renderer3d_begin_batch(r, r->current_shader,
                               &r->current_view,
                               &r->current_projection);
    }

    float tex_index = 0.0f;
    for (uint32_t i = 1; i < r->texture_slot_index; i++)
    {
        if (r->texture_slots[i] == texture_id)
        {
            tex_index = (float)i;
            break;
        }
    }

    if (tex_index == 0.0f)
    {
        if (r->texture_slot_index >= MAX_TEXTURES)
        {
            renderer3d_end_batch(r);
            renderer3d_begin_batch(r, r->current_shader,
                                   &r->current_view,
                                   &r->current_projection);
        }
        tex_index = (float)r->texture_slot_index;
        r->texture_slots[r->texture_slot_index++] = texture_id;
    }

    float transform[16];
    build_transform(transform, position, rotation, size.x, size.y);

    vec2 tex_coords[4] = {
        { 0.0f, 0.0f }, { 1.0f, 0.0f },
        { 1.0f, 1.0f }, { 0.0f, 1.0f }
    };

    for (int i = 0; i < 4; i++)
    {
        r->vertex_buffer_ptr->position  = transform_vec4(transform, r->quad_vertex_positions[i]);
        r->vertex_buffer_ptr->color     = color;
        r->vertex_buffer_ptr->tex_coord = tex_coords[i];
        r->vertex_buffer_ptr->normal = normal;
        r->vertex_buffer_ptr->tex_index = tex_index;
        r->vertex_buffer_ptr++;
    }

    r->index_count += 6;
}

void renderer3d_draw_mesh(mesh_t* mesh, GLuint shader_id, mat4* model,
                           mat4* view, mat4* projection,
                           texture_t* texture)
{
    glUseProgram(shader_id);
    set_uniform_mat4(shader_id, "u_model",      model);
    set_uniform_mat4(shader_id, "u_view",       view);
    set_uniform_mat4(shader_id, "u_projection", projection);
    set_uniform_vec3(shader_id, "u_light_pos",   MAIN_LIGHT_POS);
    set_uniform_vec3(shader_id, "u_light_color", MAIN_LIGHT_COLOR);

    if (texture && texture->id != 0)
    {
        texture_bind(texture, 0);
        glUniform1i(uniform_loc(shader_id, "u_texture"),     0);
        glUniform1i(uniform_loc(shader_id, "u_use_texture"), 1);
    }
    else
    {
        glUniform1i(uniform_loc(shader_id, "u_use_texture"), 0);
    }

    mesh_draw(mesh);
}

// ------------------------------------------------------------------
// vec3
// ------------------------------------------------------------------

vec3 vec3_zero()
{
    return (vec3){ 0.0, 0.0, 0.0 };
}

vec3 vec3_right()
{
    return (vec3){ 1.0, 0.0, 0.0 };
}

vec3 vec3_up()
{
    return (vec3){ 0.0, 1.0, 0.0 };
}

vec3 vec3_forward()
{
    return (vec3){ 0.0, 0.0, 1.0 };
}

vec3 vec3_one()
{
    return (vec3){ 1.0, 1.0, 1.0 };
}

vec3 vec3_add(vec3* a, vec3* b)
{
    return (vec3){ a->x+b->x, a->y+b->y, a->z+b->z };
}

vec3 vec3_subtract(vec3* a, vec3* b)
{
    return (vec3){ a->x-b->x, a->y-b->y, a->z-b->z };
}

vec3 vec3_multiply_scalar(vec3* vec, float scalar)
{
    return (vec3){vec->x * scalar, vec->y * scalar, vec->z * scalar};
}

vec3 vec3_cross(vec3* a, vec3* b)
{
    return (vec3){
        a->y*b->z - a->z*b->y,
        a->z*b->x - a->x*b->z,
        a->x*b->y - a->y*b->x
    };
}

float vec3_dot(vec3* a, vec3* b)
{
    return a->x*b->x + a->y*b->y + a->z*b->z;
}

// ------------------------------------------------------------------
// mat4
// ------------------------------------------------------------------

// FIX: usa designated initializer .m
mat4 mat4_identity()
{
    return (mat4){ .m = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    }};
}

// FIX: corrigido para column-major
mat4 mat4_multiply(mat4* a, mat4* b)
{
    mat4 result = {0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                result.m[j * 4 + i] += a->m[k * 4 + i] * b->m[j * 4 + k];
    return result;
}

// FIX: índices corretos para column-major (12,13,14)
mat4 mat4_translate(mat4* mat, vec3* vec)
{
    mat4 result = *mat;
    result.m[12] += vec->x;
    result.m[13] += vec->y;
    result.m[14] += vec->z;
    return result;
}

// FIX: implementado (antes retornava UB)
mat4 mat4_rotate(mat4* mat, vec3* axis, float angle)
{
    float x = axis->x,
          y = axis->y,
          z = axis->z;
    
    float len = sqrtf(x*x + y*y + z*z);
    x /= len;
    y /= len;
    z /= len;

    float c = cosf(angle),
          s = sinf(angle),
          t = 1.0f - c;

    float r[16];
    mat4_identity_raw(r);

    r[0]  = t*x*x + c;
    r[4]  = t*x*y - s*z;
    r[8]  = t*x*z + s*y;
    r[1]  = t*x*y + s*z;
    r[5]  = t*y*y + c;
    r[9]  = t*y*z - s*x;
    r[2]  = t*x*z - s*y;
    r[6]  = t*y*z + s*x;
    r[10] = t*z*z + c;

    mat4 rot = {0};
    memcpy(rot.m, r, 64);
    return mat4_multiply(mat, &rot);
}

mat4 mat4_rotate_x(mat4* mat, float angle)
{
    mat4 rot = mat4_identity();
    rot.m[5] =  cosf(angle);
    rot.m[6] =  sinf(angle);
    rot.m[9] = -sinf(angle);
    rot.m[10] =  cosf(angle);
    return mat4_multiply(mat, &rot);
}

mat4 mat4_rotate_y(mat4* mat, float angle)
{
    mat4 rot = mat4_identity();
    rot.m[0] =  cosf(angle);
    rot.m[2] = -sinf(angle);
    rot.m[8] =  sinf(angle);
    rot.m[10] =  cosf(angle);
    return mat4_multiply(mat, &rot);
}

mat4 mat4_rotate_z(mat4* mat, float angle)
{
    mat4 rot = mat4_identity();
    rot.m[0] =  cosf(angle);
    rot.m[1] =  sinf(angle);
    rot.m[4] = -sinf(angle);
    rot.m[5] =  cosf(angle);
    return mat4_multiply(mat, &rot);
}

// FIX: implementado (antes retornava UB)
mat4 mat4_scale(mat4* mat, vec3* vec)
{
    mat4 result = *mat;
    result.m[0]  *= vec->x;
    result.m[5]  *= vec->y;
    result.m[10] *= vec->z;
    return result;
}

void mat4_print(mat4* mat)
{
    for (int i = 0; i < 4; i++)
    {
        printf("[ ");
        for (int j = 0; j < 4; j++)
            printf("%8.3f ", mat->m[i * 4 + j]);
        printf("]\n");
    }
}

mat4 mat4_perspective(float fov, float aspect,
                      float near, float far)
{
    mat4 result;
    memset(result.m, 0, 16 * sizeof(float));

    float f = 1.0f / tanf(fov * 0.5f);

    result.m[0]  = f / aspect;
    result.m[5]  = f;
    result.m[10] = -(far + near) / (far - near);
    result.m[11] = -1.0f;
    result.m[14] = -(2.0f * far * near) / (far - near);

    return result;
}

mat4 mat4_look_at(vec3 eye, vec3 center, vec3 up)
{
    mat4 result;
    memset(result.m, 0, 16 * sizeof(float));

    vec3 f =
    {
        center.x-eye.x,
        center.y-eye.y,
        center.z-eye.z 
    };

    float fl = sqrtf(f.x*f.x + f.y*f.y + f.z*f.z);
    f.x /= fl; f.y /= fl; f.z /= fl;

    vec3 s =
    {
        f.y*up.z - f.z*up.y,
        f.z*up.x - f.x*up.z,
        f.x*up.y - f.y*up.x
    };
    float sl = sqrtf(s.x*s.x + s.y*s.y + s.z*s.z);
    s.x /= sl; s.y /= sl; s.z /= sl;

    vec3 u =
    {
        s.y*f.z - s.z*f.y,
        s.z*f.x - s.x*f.z,
        s.x*f.y - s.y*f.x
    };

    result.m[0]  =  s.x;
    result.m[4]  =  s.y;
    result.m[8]  =  s.z;
    result.m[1]  =  u.x;
    result.m[5]  =  u.y;
    result.m[9]  =  u.z;
    result.m[2]  = -f.x;
    result.m[6]  = -f.y;
    result.m[10] = -f.z;
    result.m[12] = -(s.x*eye.x + s.y*eye.y + s.z*eye.z);
    result.m[13] = -(u.x*eye.x + u.y*eye.y + u.z*eye.z);
    result.m[14] =  (f.x*eye.x + f.y*eye.y + f.z*eye.z);
    result.m[15] =  1.0f;

    return result;
}

// ------------------------------------------------------------------
// camera
// ------------------------------------------------------------------

const vec3 CAMERA_UP = (vec3){0, 1, 0};
bool camera_free = false;

typedef struct
{
    vec3 position;
    vec3 last_position; // usado para reverter o 'free movement' da camera

    float yaw;
    float last_yaw;

    float pitch;
    float last_pitch;

    float speed;
    float sensitivity;

} camera;

vec3 camera_get_forward(camera* cam)
{
    vec3 dir;

    dir.x = cosf(cam->yaw) * cosf(cam->pitch);
    dir.y = sinf(cam->pitch);
    dir.z = sinf(cam->yaw) * cosf(cam->pitch);

    float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    dir.x /= len; dir.y /= len; dir.z /= len;

    return dir;
}

vec3 camera_get_right(camera* cam)
{
    vec3 forward = camera_get_forward(cam);
    vec3 right = vec3_cross(&forward, (vec3*)&CAMERA_UP);

    float len = sqrtf(right.x*right.x + right.y*right.y + right.z*right.z);
    right.x /= len; right.y /= len; right.z /= len;

    return right;
}

// ------------------------------------------------------------------
// game objects
// ------------------------------------------------------------------

typedef struct
{
    vec3 position;
    vec3 rotation;
    vec3 scale;
} transform_t;

transform_t transform_identity()
{
    transform_t transform;
    transform.position = vec3_zero();
    transform.rotation = vec3_zero();
    transform.scale    = vec3_one();
    return transform;
}

typedef struct
{
    transform_t transform;
    vec3        speed;
    mesh_t*     mesh;
    texture_t*  texture;
} game_object_t;

game_object_t game_object_create()
{
    game_object_t object;
    object.transform = transform_identity();
    object.speed = vec3_zero();
    return object;
}

void game_object_update(game_object_t* object, float dt)
{
    vec3 delta = vec3_multiply_scalar(&object->speed, dt);
    object->transform.position = vec3_add(
        &object->transform.position,
        &delta
    );
}



// ------------------------------------------------------------------
// game world
// ------------------------------------------------------------------

#define MAX_GAME_OBJECTS 1024

typedef struct
{
    game_object_t objects[MAX_GAME_OBJECTS];
    bool          active[MAX_GAME_OBJECTS];
    int           count;
} game_world_t;

void game_world_init(game_world_t* world)
{
    memset(world, 0, sizeof(game_world_t));
}

// retorna -1 se lotado
int game_world_add(game_world_t* world, game_object_t object)
{
    for (int i = 0; i < MAX_GAME_OBJECTS; i++)
    {
        if (!world->active[i])
        {
            world->objects[i] = object;
            world->active[i]  = true;
            world->count++;
            return i;
        }
    }
    printf("game_world: sem slots disponíveis\n");
    return -1;
}

void game_world_remove(game_world_t* world, int index)
{
    if (index < 0 || index >= MAX_GAME_OBJECTS)
        return;
    
    if (!world->active[index])
        return;

    world->active[index] = false;
    world->count--;
}

void game_world_update(game_world_t* world, float dt)
{
    for (int i = 0; i < MAX_GAME_OBJECTS; i++)
    {
        if (!world->active[i]) continue;
        game_object_update(&world->objects[i], dt);
    }
}

void game_world_render(game_world_t* world, GLuint shader,
                       mat4* view, mat4* projection)
{
    for (int i = 0; i < MAX_GAME_OBJECTS; i++)
    {
        if (!world->active[i]) continue;
        game_object_t* obj = &world->objects[i];
        if (!obj->mesh) continue;

        mat4 model = mat4_identity();
        model = mat4_translate(&model, &obj->transform.position);
        model = mat4_rotate_y(&model,   obj->transform.rotation.y);
        model = mat4_scale   (&model,  &obj->transform.scale);

        renderer3d_draw_mesh(obj->mesh, shader, &model,
                             view, projection, obj->texture);
    }
}

// ------------------------------------------------------------------
// callbacks
// ------------------------------------------------------------------

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------

const int   WINDOW_WIDTH  = 1366;
const int   WINDOW_HEIGHT = 768;
const char* WINDOW_TITLE  = "Game FOG";

int main()
{
    if (!glfwInit())
    {
        printf("Erro ao inicializar GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, NULL, NULL);
    if (!window)
    {
        printf("Erro ao criar janela\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        printf("Erro ao inicializar GLAD\n");
        return -1;
    }

    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // FIX: depth test habilitado
    glEnable(GL_DEPTH_TEST);

    time_init();
    input_init(window);

    shader_t shader = shader_create_from_src(vertex_src, fragment_src);

    vertex3d_t cube_verts[] =
    {
        // Frente (+Z)
        {{-0.5,-0.5, 0.5}, {1,0,0,1}, {0,0}, {0,0,1}, 0},
        {{ 0.5,-0.5, 0.5}, {1,0,0,1}, {1,0}, {0,0,1}, 0},
        {{ 0.5, 0.5, 0.5}, {1,0,0,1}, {1,1}, {0,0,1}, 0},
        {{-0.5, 0.5, 0.5}, {1,0,0,1}, {0,1}, {0,0,1}, 0},

        // Trás (-Z)
        {{-0.5,-0.5,-0.5}, {1,0,0,1}, {0,0}, {0,0,-1}, 0},
        {{ 0.5,-0.5,-0.5}, {1,0,0,1}, {1,0}, {0,0,-1}, 0},
        {{ 0.5, 0.5,-0.5}, {1,0,0,1}, {1,1}, {0,0,-1}, 0},
        {{-0.5, 0.5,-0.5}, {1,0,0,1}, {0,1}, {0,0,-1}, 0},

        // Esquerda (-X)
        {{-0.5,-0.5,-0.5}, {1,0,0,1}, {0,0}, {-1,0,0}, 0},
        {{-0.5,-0.5, 0.5}, {1,0,0,1}, {1,0}, {-1,0,0}, 0},
        {{-0.5, 0.5, 0.5}, {1,0,0,1}, {1,1}, {-1,0,0}, 0},
        {{-0.5, 0.5,-0.5}, {1,0,0,1}, {0,1}, {-1,0,0}, 0},

        // Direita (+X)
        {{ 0.5,-0.5,-0.5}, {1,0,0,1}, {0,0}, {1,0,0}, 0},
        {{ 0.5,-0.5, 0.5}, {1,0,0,1}, {1,0}, {1,0,0}, 0},
        {{ 0.5, 0.5, 0.5}, {1,0,0,1}, {1,1}, {1,0,0}, 0},
        {{ 0.5, 0.5,-0.5}, {1,0,0,1}, {0,1}, {1,0,0}, 0},

        // Baixo (-Y)
        {{-0.5,-0.5,-0.5}, {1,0,0,1}, {0,0}, {0,-1,0}, 0},
        {{ 0.5,-0.5,-0.5}, {1,0,0,1}, {1,0}, {0,-1,0}, 0},
        {{ 0.5,-0.5, 0.5}, {1,0,0,1}, {1,1}, {0,-1,0}, 0},
        {{-0.5,-0.5, 0.5}, {1,0,0,1}, {0,1}, {0,-1,0}, 0},

        // Cima (+Y)
        {{-0.5, 0.5,-0.5}, {1,0,0,1}, {0,0}, {0,1,0}, 0},
        {{ 0.5, 0.5,-0.5}, {1,0,0,1}, {1,0}, {0,1,0}, 0},
        {{ 0.5, 0.5, 0.5}, {1,0,0,1}, {1,1}, {0,1,0}, 0},
        {{-0.5, 0.5, 0.5}, {1,0,0,1}, {0,1}, {0,1,0}, 0},
    };

    uint32_t cube_indices[] =
    {
        0,1,2,   2,3,0,     // Frente    (verts 0-3)
        4,5,6,   6,7,4,     // Trás      (verts 4-7)
        8,9,10,  10,11,8,   // Esquerda  (verts 8-11)
        12,13,14, 14,15,12, // Direita   (verts 12-15)
        16,17,18, 18,19,16, // Baixo     (verts 16-19)
        20,21,22, 22,23,20  // Cima      (verts 20-23)
    };

    mesh_t cube = mesh_create(cube_verts, 24, cube_indices, 36);

    renderer3d_t renderer;
    renderer3d_init(&renderer);

    mat4 projection = mat4_perspective(PI / 3.0f, (float)WINDOW_WIDTH / WINDOW_HEIGHT, 0.1f, 100.0f);
    mat4 view = (mat4){0};

    float cube_angle = 0.0f;

    camera camera;

    camera.position = (vec3){0, 2.0f, 3.0f};
    camera.last_position = camera.position;

    camera.yaw = -PI / 2.0f;
    camera.last_yaw = camera.yaw;

    camera.pitch = -PI / 8;
    camera.last_pitch = camera.pitch;

    camera.speed = 5.0f;
    camera.sensitivity = 0.002f;

    obj_data_t cow_data = {0};
    load_obj("cow.obj", &cow_data);

    int total_verts = cow_data.face_count * 3;

    vertex3d_t* cow_verts  = malloc(total_verts * sizeof(vertex3d_t));
    uint32_t*   cow_idx    = malloc(total_verts * sizeof(uint32_t));

    for (int i = 0; i < total_verts; i++)
    {
        int pi = cow_data.pos_indices[i];
        int ti = cow_data.tex_indices[i];
        int ni = cow_data.nor_indices[i];

        cow_verts[i].position  = cow_data.positions[pi];
        cow_verts[i].tex_coord = cow_data.texcoords[ti];
        cow_verts[i].normal    = cow_data.normals[ni];
        cow_verts[i].color     = (vec4){1.0f, 0.8f, 0.6f, 1.0f};
        cow_verts[i].tex_index = 0.0f;

        cow_idx[i] = (uint32_t)i; // cada vértice é único
    }

    mesh_t cow = mesh_create(cow_verts, total_verts, cow_idx, total_verts);

    free(cow_verts);
    free(cow_idx);
    obj_data_free(&cow_data);

    texture_t color_map_texture = texture_load("colormap.png");

    game_world_t world;
    game_world_init(&world);

    // cria uma vaca
    game_object_t object = game_object_create();
    object.mesh    = &cow;
    object.texture = &color_map_texture;
    int object_id = game_world_add(&world, object);

    world.objects[object_id].speed = (vec3){ 1.0f, 0.0f, 0.0f };

    while (!glfwWindowShouldClose(window))
    {
        time_update();
        input_update();

        if (input_get_key(GLFW_KEY_ESCAPE))
            glfwSetWindowShouldClose(window, 1);

        world.objects[object_id].transform.rotation.y = cube_angle;
        world.objects[object_id].speed = (vec3){ sinf(time_total()), 0.0, 0.0 };
        game_world_update(&world, time_delta());
        
        cube_angle += 1.0f * time_delta();

        vec3 forward = camera_get_forward(&camera);
        vec3 right   = camera_get_right(&camera);

        if (input_get_key_down(GLFW_KEY_TAB))
        {
            camera_free = !camera_free;

            if (!camera_free)
            {
                camera.position = camera.last_position;
                camera.yaw = camera.last_yaw;
                camera.pitch = camera.last_pitch;
            }
            else
            {
                camera.last_position = camera.position;
                camera.last_yaw = camera.yaw;
                camera.last_pitch = camera.pitch;
            }
        }

        if (camera_free)
        {
            static double last_x = 0, last_y = 0;
            static int first_mouse = 1;

            double mouse_x, mouse_y;
            input_get_mouse_position(&mouse_x, &mouse_y);

            if (first_mouse)
            {
                last_x = mouse_x;
                last_y = mouse_y;
                first_mouse = 0;
            }

            float dx = mouse_x - last_x;
            float dy = last_y - mouse_y;

            last_x = mouse_x;
            last_y = mouse_y;

            dx *= camera.sensitivity;
            dy *= camera.sensitivity;

            camera.yaw   += dx;
            camera.pitch += dy;

            // clamp pitch
            if (camera.pitch >  1.5f) camera.pitch =  1.5f;
            if (camera.pitch < -1.5f) camera.pitch = -1.5f;

            float velocity = camera.speed * time_delta();

            if (input_get_key(GLFW_KEY_W))
                camera.position = vec3_add(&camera.position,
                    &(vec3){forward.x * velocity, forward.y * velocity, forward.z * velocity});

            if (input_get_key(GLFW_KEY_S))
                camera.position = vec3_subtract(&camera.position,
                    &(vec3){forward.x * velocity, forward.y * velocity, forward.z * velocity});

            if (input_get_key(GLFW_KEY_A))
                camera.position = vec3_subtract(&camera.position,
                    &(vec3){right.x * velocity, right.y * velocity, right.z * velocity});

            if (input_get_key(GLFW_KEY_D))
                camera.position = vec3_add(&camera.position,
                    &(vec3){right.x * velocity, right.y * velocity, right.z * velocity});

            if (input_get_key(GLFW_KEY_Q))
                camera.position = vec3_add(&camera.position, &(vec3){0.0, -velocity, 0.0 });
            
            if (input_get_key(GLFW_KEY_E))
                camera.position = vec3_add(&camera.position, &(vec3){0.0, velocity, 0.0});
        }
        
        vec3 target = vec3_add(&camera.position, &forward);

        view = mat4_look_at(camera.position, target, CAMERA_UP);

        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // cubo girando em Y
        mat4 model = mat4_identity();
        
        
        model = mat4_translate(&model, &(vec3){ 0.0, 1.0, 0.0 });
        glUseProgram(shader.id);

        //  renderer3d_draw_mesh(&cube, shader.id, &model, &view, &projection);

        game_world_render(&world, shader.id, &view, &projection);

        mat4 identity = mat4_identity();

        shader_set_mat4(&shader, "u_model", &identity);
        shader_set_mat4(&shader, "u_view", &view);
        shader_set_mat4(&shader, "u_projection", &projection);

        renderer3d_begin_batch(&renderer, shader.id, &view, &projection);


        renderer3d_end_batch(&renderer);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    texture_destroy(&color_map_texture);
    mesh_destroy(&cube);
    mesh_destroy(&cow);

    game_world_remove(&world, object_id);
    renderer3d_destroy(&renderer);
    
    glfwTerminate();
    return 0;
}