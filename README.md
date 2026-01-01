# Instalación de Ollama para iGPU AMD en Gentoo

## Especificaciones del sistema (ThinkPad T14s)
- **CPU**: AMD Ryzen 7 PRO 4750U (16 hilos) - Bueno para inferencia por CPU
- **GPU**: AMD Radeon Vega Mobile Series - **Punto clave para aceleración**
- **RAM**: 32GB - Excelente para modelos grandes

Con el comando `lspci | grep -i vga` se puede obtener informacion de la familia de la iGPU, y con el comando `rocminfo` nos daria el match exacto, pero en mi caso para poder instalar rocminfo, tendria primero que compilar rocm, asi que no es muy conveniente. 

De cualquier manera se puede checar mas detalladamente y buscar de forma mas minusiosa la arquitectura correcta en el enlace "Documentacion oficial de AMDGPU/LLVM", en mi caso la mia es gfx90c.

Dentro de la documentacion de ROCm, mientras buscaba mas informacion sobre mi arquitectura, pude ver que las instrucciones para ambas architecutras son similares (gfx90c y gfx900 comparten sintaxis), y si bien gfx90c no esta dentro del soporte oficial, gfx900 si, por lo tanto decidi compilar ROCm apuntando especificamente hacia gfx900.

**Recursos útiles**:
- [Documentación oficial de AMDGPU/LLVM](https://llvm.org/docs/AMDGPUUsage.html) - Para buscar arquitecturas específicas
- [Documentación ROCm para gfx900](https://rocmdocs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX900.html) - Sintaxis compatible con gfx90c

### Investigación previa
En el articulo "Running Ollama on AMD iGPU", se puede ver como es posible correr en iGPU con un backend diferente a los soportados oficialmente, y aunque este articulo fue pensado para ollama con docker, mi objetivo es lograr una instalacion mas basada en codigo fuente buscando maxima optimizacion. 

**Referencias**:
- [Video tutorial: Ollama + AMD iGPU](https://www.youtube.com/watch?v=G-kpvlvKM1g)
- [Artículo técnico: Running Ollama on AMD iGPU](https://blog.machinezoo.com/Running_Ollama_on_AMD_iGPU)

## Primer paso en Gentoo: Preparación de ROCm

**Próximos pasos** (continuará en la siguiente parte):
1. Configurar USE flags para ROCm en Gentoo
2. Compilar componentes esenciales del stack ROCm
3. Verificar compatibilidad con `gfx90c`
4. Integrar con Ollama

### De CUDA a ROCm: Habilitando LLMs en tu iGPU con Ollama

Para entender cómo Ollama puede ejecutar modelos de lenguaje en tu iGPU AMD/Radeon, primero es crucial comprender el contexto del desarrollo del software para GPUs en el ámbito de la IA.

**1. El Liderazgo de NVIDIA y el Ecosistema CUDA**

NVIDIA lanzó **CUDA en 2006**, mucho antes del "boom" de la IA actual. Su visión era transformar las GPUs, que son excelentes en realizar miles de cálculos sencillos de forma paralela, de meras tarjetas gráficas a "procesadores de propósito general" para cómputo de alto rendimiento.

CUDA fue la herramienta perfecta en el momento justo cuando el Deep Learning comenzó a despegar años después. NVIDIA ya tenía la infraestructura, las herramientas y, lo más importante, un **ecosistema de desarrolladores robusto** familiarizado con el cómputo en GPU. Esto les dio una ventaja masiva, convirtiendo a CUDA en el estándar de facto para la investigación y el desarrollo de IA. Por ello, la gran mayoría de los modelos y frameworks de IA se optimizaron inicialmente para las GPUs NVIDIA y CUDA.

**2. La Solución de AMD: ROCm, HIP y la Inferencia en iGPUs**

Para que AMD pueda competir y habilitar la inferencia de LLMs en sus GPUs (incluyendo las iGPUs) con plataformas como Ollama, necesita un "motor de inferencia" que maneje el cómputo y, crucialmente, que pueda interactuar con el vasto código existente escrito para CUDA.

*   **ROCm (Radeon Open Compute platform):** Es la plataforma de código abierto de AMD para cómputo de alto rendimiento. Permite que las GPUs de AMD realicen los cálculos tensoriales masivos requeridos por los LLMs, yendo más allá de sus funciones gráficas tradicionales. Es la base sobre la que se construyen las capacidades de IA de AMD.
*   **HIP (Heterogeneous-Compute Interface for Portability):** Dado que la mayoría del código de IA está escrito para CUDA, HIP es una capa de portabilidad crítica. Actúa como un "traductor", permitiendo que el código de cómputo basado en CUDA se ejecute en arquitecturas AMD (como GCN 5.1) con mínimas o ninguna modificación. Esto evita la necesidad de reescribir modelos enteros desde cero para AMD.

**3. El Desafío y la Solución de la iGPU: Memoria Compartida y GPU Offloading**

La particularidad de una iGPU es que, a diferencia de una GPU dedicada, **no tiene VRAM física propia**. En su lugar, utiliza una **Arquitectura de Memoria Unificada (UMA)**. Esto significa que la iGPU comparte la RAM del sistema con el CPU, y el sistema operativo asigna dinámicamente o de forma fija una porción de esta RAM para que funcione como su memoria de video (VRAM).

*   **GPU Offloading (Descarga de Cómputo):** Es el concepto fundamental. Consiste en mover estratégicamente las capas del modelo de lenguaje (o partes de ellas) desde la RAM principal hacia la porción de RAM que la iGPU utiliza como su memoria. Esto permite que la iGPU, con su arquitectura paralela, maneje la carga computacional intensiva.
*   **Unified Memory (en ROCm):** Esta característica es clave para las iGPUs, ya que aprovecha la UMA subyacente. Permite que el sistema y ROCm gestionen de manera eficiente esta memoria compartida, facilitando el *offload* de grandes volúmenes de datos del modelo a la iGPU.
*   **OpenMP:** Mientras la iGPU se encarga de la carga principal, el CPU sigue desempeñando un papel vital. OpenMP gestiona la paralelización eficiente de las tareas en los núcleos del procesador, preparando y coordinando los datos antes de su envío a la GPU, especialmente en este entorno de memoria compartida.

**Optimización Práctica para iGPUs (Especialmente en Laptops):**

Debido a la naturaleza de la UMA, la cantidad de RAM del sistema que una iGPU puede usar como VRAM es un factor crítico. En muchos sistemas (particularmente laptops), la asignación predeterminada puede ser limitada. Una optimización común y efectiva es **aumentar la cantidad de RAM asignada a la iGPU como VRAM desde la configuración de la BIOS/UEFI** del sistema. Esto permite cargar modelos más grandes o más capas de un modelo en la "VRAM" de la iGPU, mejorando significativamente el rendimiento.

**El Resultado:**

Gracias a ROCm, HIP, el GPU Offloading eficiente y una gestión adecuada de la memoria compartida (a menudo optimizada manualmente), las iGPUs AMD pueden procesar los tokens de los LLMs significativamente más rápido que los núcleos seriales de la CPU. A pesar de compartir el bus de memoria, la arquitectura paralela masiva de la iGPU elimina los cuellos de botella, logrando que la generación de texto sea fluida y receptiva en Ollama, democratizando el acceso a la IA en hardware más común.

### Configuración inicial

Lo primero que haré será modificar los archivos `package.use` y `package.accept_keywords` para habilitar las opciones necesarias en los paquetes relacionados con LLVM y OpenMP:

```conf
llvm-runtimes/openmp ompt debug offload llvm_targets_AMDGPU
llvm-runtimes/offload ompt offload llvm_targets_AMDGPU
llvm-runtimes/clang-runtime omtp offload
```

En mi caso, la opción `llvm_targets_AMDGPU` no fue reconocida, lo cual es crucial ya que sin ella el código se compila pero no permite interactuar con la iGPU. Para solucionarlo, creé un nuevo archivo en `/etc/portage/profile/package.use.mask/ollama` con el siguiente contenido:

```conf
llvm-runtimes/offload -llvm_targets_AMDGPU
llvm-runtimes/openmp -llvm_targets_AMDGPU
```

Con esto resuelto, procedí a instalar los paquetes necesarios:

```bash
emerge -av llvm-runtimes/openmp llvm-runtimes/offload llvm-runtimes/clang-runtime llvm-core/clang
```

### Configuración de ROCm

Para ROCm, actualicé `package.accept_keywords` con las siguientes entradas:

```conf
app-misc/ollama ~amd64
dev-libs/roc* ~amd64
dev-util/rocm* ~amd64
dev-util/hip* ~amd64
sci-libs/hip* ~amd64
sci-libs/roc* ~amd64
dev-util/hip ~amd64
dev-build/rocm-cmake ~amd64
dev-util/Tensile ~amd64
```

Dado que tengo `/var/tmp/portage` montado en `tmpfs`, excluí ciertos paquetes de la memoria temporal para evitar problemas durante la compilación:

```conf
llvm-core/clang notmpfs.conf
dev-util/Tensile notmpfs.conf
sci-libs/rocBLAS notmpfs.conf
dev-libs/rocm-device-libs notmpfs.conf
dev-util/hipcc notmpfs.conf
dev-util/hip notmpfs.conf
sci-libs/hipBLAS notmpfs.conf
dev-libs/boost normpfs.conf
llvm-core/lld notmpfs.conf
```

El archivo `notmpfs.conf` se configura de la siguiente manera:

```conf
# path /etc/portage/env/notmpfs.conf
PORTAGE_TMPDIR="/var/tmp/notmpfs"
```

### Definición del objetivo de compilación

A continuación, establecí `gfx900` como objetivo de compilación para AMDGPU:

```conf
# path /etc/portage/package.use/00-amdgpu-targets
*/* AMDGPU_TARGETS: -* gfx900
```

### Compilación de ROCm

Una vez configurado todo, procedí a compilar ROCm. En mi caso, utilicé el siguiente comando tras analizar la ebuild correspondiente, aunque también incluyo una alternativa más general:

```bash
emerge -av @rocm
# emerge -av rocm-libs/rocm-core
```

O, si se prefiere una instalación explícita:

```bash
emerge -av rocm-hip rocm-opencl-runtime rocm-smi
```

### Verificación del funcionamiento de OpenMP y Offload

Siguiendo los pasos de la wiki de Gentoo sobre ROCm, procedí a verificar que el offload de OpenMP esté funcionando correctamente. Para ello, desarrollé un programa en C que utiliza directivas de OpenMP para ejecutar código en la GPU:

```c
#include <omp.h>
#include <stdio.h>

int main() {
  int is_cpu = 1;
  
#pragma omp target map(from : is_cpu)
  {
    is_cpu = omp_is_initial_device();
  }

  if (is_cpu) {
    printf("Fallo: El código se ejecutó en la CPU.\n");
  } else {
    printf("Éxito: ¡El código se ejecutó en la GPU AMD!\n");
  }
  return 0;
}
```

Este código emplea la función `omp_is_initial_device()`, que retorna un valor booleano: `0` indica que la ejecución se realizó en la GPU, mientras que `1` significa que se mantuvo en la CPU. Si el offload funciona correctamente, el programa imprimirá el mensaje de éxito.

Para compilarlo, utilicé las banderas recomendadas por la documentación de Gentoo, con algunas adaptaciones específicas para mi configuración:

```bash
clang -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
  --rocm-path=/usr \
  --rocm-device-lib-path=/usr/lib/amdgcn/bitcode \
  -Xopenmp-target=amdgcn-amd-amdhsa --offload-arch=gfx900 \
  test_rocm.c -o test_rocm
```

La ejecución del programa confirmó que el offload funciona adecuadamente, mostrando el mensaje esperado: "Éxito: ¡El código se ejecutó en la GPU AMD!". Esto valida que la configuración de ROCm y OpenMP está operativa y lista para el siguiente paso.

### Instalación de Ollama desde el código fuente

Antes de proceder con la compilación desde el código fuente, probé Ollama utilizando Docker y el script de instalación oficial (`curl -fsSL https://ollama.com/install.sh | sh`). Si bien ambas opciones funcionaron correctamente, solo utilizaban la CPU. El objetivo de estas pruebas era aprovechar la GPU y obtener un binario optimizado para mi hardware, aprovechando al máximo las ventajas de Gentoo.

El primer paso fue clonar el repositorio de Ollama:

```bash
git clone https://github.com/ollama/ollama.git
```

Al explorar el repositorio, noté que combina componentes en C++ y Go. Un archivo clave es `CMakePresets.json`, el cual modifiqué para especificar mi arquitectura de GPU (`gfx900`) y eliminar las demás:

```diff
diff --git a/CMakePresets.json b/CMakePresets.json
index 6fcdf4d2..140de900 100644
--- a/CMakePresets.json
+++ b/CMakePresets.json
@@ -72,8 +72,8 @@
       "name": "ROCm 6",
       "inherits": [ "ROCm" ],
       "cacheVariables": {
-        "CMAKE_HIP_FLAGS": "-parallel-jobs=4",
-        "AMDGPU_TARGETS": "gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-",
+        "CMAKE_HIP_FLAGS": "",
+          "GPU_TARGETS": "gfx900",
         "OLLAMA_RUNNER_DIR": "rocm"
       }
     },
```

Luego, procedí con la configuración y compilación:

```bash
# Verificar la configuración
cmake --preset "ROCm 6"
# Compilar los componentes de C++
cmake --build --preset "ROCm 6"
```

Esto genera las bibliotecas necesarias en `build/lib/ollama`, que luego se enlazarán con el programa principal compilado en Go:

```bash
go build -v -o build/bin/ollama .
```


Para probar el binario:

```bash
# Desde el directorio raíz del proyecto
./build/bin/ollama serve
# En otra terminal
./build/bin/ollama -h
```

Descargué un modelo para validar el funcionamiento:

```bash
./build/bin/ollama pull deepseek-coder:6.7b
```

Utilicé un script en Python para probar la inferencia y monitoreé el uso de la GPU con `rocm-smi`:

```python
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

def main():
    phrase = input("What do you want to learn today?: ")

    instructions = """
    Write a simple example of code for what the user wants to learn.
    Today they want to learn:
    {phrase}
    """

    myPrompt = PromptTemplate(
        input_variables=["phrase"],
        template=instructions
    )

    llm = OllamaLLM(
        model="deepseek-coder:1.3b",
        temperature=0.3
    )

    chain = myPrompt | llm

    print("\nResult:\n")

    for chunk in chain.stream({"phrase": phrase}):
        if chunk:
            print(chunk, end="", flush=True)

    print("\n")

if __name__ == "__main__":
    main()
```

### Configuración del servicio con OpenRC

Creé un script de inicio para OpenRC en `/etc/init.d/ollama`:

```conf
#!/sbin/openrc-run

name="ollama"
description="Ollama LLM server daemon"

command="/usr/local/bin/ollama"
command_args="serve"
command_background="yes"
pidfile="/run/${RC_SVCNAME}.pid"

start_pre() {
    export HOME=/home/oscar
    export HSA_OVERRIDE_GFX_VERSION=9.0.0
    export ROC_ENABLE_PRE_VEGA=1
    export OLLAMA_HOST=127.0.0.1:11434
}

depend() {
    need net
    after udev
}
```

Para iniciar y habilitar el servicio:

```bash
rc-service ollama start
rc-update add ollama default
```

### Consideraciones de hardware

En mi caso, la VRAM asignada por defecto a la iGPU era de 512 MB, lo cual resultaba insuficiente. Modificando la BIOS, pude aumentarla a 2 GB, permitiendo ejecutar modelos como `deepseek-coder:1.3b` o `qwen2.5-coder:1.5b` con offload a la VRAM (memoria compartida con la RAM). 

### BIOS
Al encender la laptop, cuando se muestra el logo de lenovo, con F1 se puede acceder a la BIOS, despues hay que ir hacia Config, y buscar el parametro Display, despues UMA Framebuffer size hay un menu desplegable, con las opciones para modificar la RAM que el sistema asigna como VRAM.

Los modelos que funcionan adecuadamente en mi configuración son:

| Nombre               | ID           | Tamaño   | Modificado       |
|----------------------|--------------|----------|------------------|
| qwen3:1.7b           | 8f68893c685c | 1.4 GB   | Hace una hora    |
| qwen2.5-coder:1.5b   | d7372fd82851 | 986 MB   | Hace una hora    |
| deepseek-coder:1.3b  | 3ddd2d3fc8d2 | 776 MB   | Hace 2 horas     |

-----------------------------------
Nota: Aunque no estoy seguro de si es estrictamente necesario, configuré `LD_LIBRARY_PATH` para incluir las bibliotecas compiladas:

```bash
export LD_LIBRARY_PATH=$PWD/build/lib/ollama:$LD_LIBRARY_PATH
```

### 🎉 ¡Gracias por leer!

Espero que esta guía te haya sido útil para configurar Ollama con soporte GPU en Gentoo. Si lograste hacerlo funcionar en tu sistema, ¡felicitaciones! Has aprovechado al máximo la filosofía de personalización que ofrece Gentoo.

Si encuentras algún problema o tienes sugerencias para mejorar el artículo, no dudes en dejar un comentario. ¡Compartir conocimiento es lo que hace crecer a la comunidad!

### 💖 ¿Te gustó el contenido?

Te dejo un documento detallado con todos los pasos que segui en el primer comentario **[galo_o](https://ko-fi.com/galo_o)**.  Tu apoyo me motiva a seguir creando contenido detallado y accesible para la comunidad.

### 🚀 ¡Mucha suerte en tu implementación!

Que tus modelos corran rápido y tus compilaciones sean exitosas. ¡Hasta la próxima!
