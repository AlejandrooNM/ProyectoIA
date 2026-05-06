# Análisis Neural de Tumores Cerebrales v2.0

## Descripción
Sistema modular de red neuronal para clasificación automática de tumores cerebrales y evaluación de riesgo quirúrgico basado en imágenes MRI.

## Estructura Modular

```
Proyecto IA/
├── principal.py                          # Punto de entrada principal
├── requisitos.txt                 # Dependencias del proyecto
├── LEEME.md                        # Este archivo
└── src/                             # Paquete principal
    ├── __init__.py                 # Inicializador del paquete
    ├── configuraciones.py          # Configuraciones centralizadas
    ├── funciones_activacion.py     # Funciones de activación
    ├── capas.py                   # Capas de redes neuronales
    ├── redes_neuronales.py        # Redes neuronales principales
    ├── simulador_imagen.py        # Simulador de imágenes cerebrales
    ├── cargador_dicom.py          # Cargador de imágenes DICOM
    ├── analyzer.py                 # Analizador principal integrado
    ├── visualizacion.py           # Sistema de visualización
    ├── informes.py                # Generación de reportes
    └── utilidades.py              # Funciones auxiliares
```

## Módulos

### `configuraciones.py`
Contiene todas las configuraciones centralizadas:
- Paletas de colores
- Tipos de tumores y sus propiedades
- Clases y grados WHO
- Parámetros de redes neuronales
- Patrones para datos sintéticos
- Curvas de supervivencia

**Ventajas**: 
- Cambios de configuración sin modificar código
- Fácil mantenimiento de constantes

### `funciones_activacion.py`
Funciones de activación y utilidades:
- ReLU, Sigmoid, Softmax, Tanh, Leaky ReLU
- Derivadas de funciones de activación
- Factory para obtener funciones por nombre

**Ventajas**:
- Código reutilizable
- Fácil agregar nuevas funciones de activación

### `capas.py`
Capas de redes neuronales:
- `CapaDensa`: Capa completamente conectada
- `CapaConvolucional2D`: Convolución 2D
- `CapaPooling`: Max Pooling 2D

**Ventajas**:
- Abstracción de capas
- Backpropagation optimizado con Adam
- Fácil construcción de redes

### `redes_neuronales.py`
Redes neuronales especializadas:
- `RedNeuronalClasificacionTumor`: Clasificación de tipos (5 clases)
- `RedNeuronalRiesgoQuirurgico`: Evaluación de riesgo (0-100)

**Ventajas**:
- Lógica de entrenamiento encapsulada
- Generación de datos sintéticos integrada
- Predicciones simples y claras

### `simulador_imagen.py`
Simulador de imágenes MRI cerebrales:
- Generación de cerebro base realista
- Adición de tumores con características morfológicas
- Extracción de 10 características numéricas

**Ventajas**:
- Datos de prueba sin DICOM
- Realismo anatómico
- Features normalizadas para la red

### `cargador_dicom.py`
Cargador de imágenes DICOM reales:
- Lectura de series DICOM
- Normalización de volúmenes
- Segmentación automática
- Cálculo de métricas clínicas
- Extracción de features

**Ventajas**:
- Soporte para datos reales
- Compatibilidad flexible
- Manejo de errores robusto

### `analyzer.py`
Analizador principal integrado:
- Coordinación de redes neuronales
- Análisis completo de tumores
- Cálculo de pronósticos
- Generación de recomendaciones

**Ventajas**:
- Orquestación de componentes
- Interfaz unificada
- Resultado estructurado

### `visualizacion.py`
Sistema de visualización:
- Reportes visuales profesionales
- Gráficos especializados (gauge, radar, curvas)
- Colores médicos consistentes
- Exportación a PNG

**Ventajas**:
- Visualización completa en un solo reporte
- Estilos médicos profesionales
- Fácil de personalizar

### `informes.py`
Generación de reportes textuales:
- Reporte clínico estructurado en consola
- Información organizada
- Avisos legales

**Ventajas**:
- Salida formateada
- Fácil lectura
- Compatible con logs

### `utilidades.py`
Funciones auxiliares:
- Entrada de datos del usuario
- Validación de rutas
- Gestión de directorios

**Ventajas**:
- Funciones reutilizables
- Separación de concerns
- Mantenibilidad mejorada

## Uso

### Instalación
```bash
pip install -r requisitos.txt
```

### Ejecución
```bash
python principal.py
```

### Opciones
1. **Modo DICOM**: Analizar imágenes reales (proporcionar ruta de carpeta con archivos .dcm)
2. **Modo Simulado**: Demostración con imágenes sintéticas (sin dependencias DICOM)

## Ventajas de la Estructura Modular

1. **Mantenibilidad**: Cada módulo tiene responsabilidad única
2. **Escalabilidad**: Fácil agregar nuevas redes o características
3. **Testabilidad**: Cada módulo puede testearse independientemente
4. **Reutilización**: Los módulos pueden usarse en otros proyectos
5. **Documentación**: Cada módulo está auto-documentado
6. **Flexibilidad**: Cambios sin afectar otros módulos
7. **Importación**: Importar funcionalidades específicas

## Ejemplo de uso modular

```python
from src.redes_neuronales import RedNeuronalClasificacionTumor
from src.simulador_imagen import SimuladorImagenCerebral

# Crear red
red = RedNeuronalClasificacionTumor()
X_train, y_train = red.generar_datos_sinteticos()
red.entrenar(X_train, y_train)

# Simular imagen
img_base = SimuladorImagenCerebral.generar_cerebro_base()
img_tumor, pos = SimuladorImagenCerebral.agregar_tumor(img_base)
features = SimuladorImagenCerebral.extraer_features(img_tumor, pos)

# Predecir
prediccion = red.predecir(features)
```

## Dependencias

- **numpy**: Computación numérica
- **matplotlib**: Visualización
- **scipy**: Procesamiento científico
- **scikit-image**: Procesamiento de imágenes
- **pydicom**: Lectura de archivos DICOM (opcional)
- **opencv-python**: Visión por computadora (opcional)

## Autor
Equipo de Neurooncología

## Versión
2.0 - Modular

## Licencia
Para uso exclusivo de profesionales médicos

---

**Nota**: Este es un sistema de apoyo a la decisión. No reemplaza la evaluación clínica de especialistas.
