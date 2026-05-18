// stb_vorbis_impl.c
// Compilado separadamente, sem poluição de macros

// Define as configurações desejadas
#define STB_VORBIS_NO_STDIO   // se não usar FILE*
#define STB_VORBIS_NO_PULLDATA_API  // se não precisar

// Implementação principal
#define STB_VORBIS_IMPLEMENTATION
#include "stb_vorbis.h"   // nota: .h, não .c