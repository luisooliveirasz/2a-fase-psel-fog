# 🐍 Cube Snake

Feito para a segunda fase do processo seletivo do grupo de extensão Fellowship of the Gaming do ICMC-USP.

---

<div align="center">

![C](https://img.shields.io/badge/C-00599C?style=for-the-badge&logo=c&logoColor=white)
![OpenGL](https://img.shields.io/badge/OpenGL-5586A4?style=for-the-badge&logo=opengl&logoColor=white)
![GLFW](https://img.shields.io/badge/GLFW-black?style=for-the-badge&logoColor=white)
![GLAD](https://img.shields.io/badge/GLAD-orange?style=for-the-badge&logoColor=white)

</div>

---

## 🛠️ Tecnologias

| Tecnologia | Papel no projeto |
|---|---|
| **C** | Linguagem principal - toda a lógica, matemática e estruturas de dados |
| **OpenGL 3.3 Core** | API gráfica — renderização de meshes, shaders, texturas |
| **GLFW** | Criação de janela, contexto OpenGL e captura de input |
| **GLAD** | Loader das extensões OpenGL |
| **stb_image** | Carregamento de texturas (`.jpg`, `.png`) |
| **stb_truetype** | carregamento de fontes (`.ttf`) |
| **miniaudio** | carregamento de sons/músicas (`.wav`) |

## Build (em linux) a partir do código fonte:
```bash
cd build
cmake ..
make
```
(Necessário ter o glfw e cmake instalados)

## Assets utilizados:
https://sketchfab.com/3d-models/apple-low-poly-76k-9mb-2k-626e626482fd431aa67b685b92fc5fbf
https://opengameart.org/content/4-chiptunes-adventure
https://opengameart.org/content/impact
https://opengameart.org/content/menu-selection-click