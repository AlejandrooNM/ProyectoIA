"""
Paquete de Análisis de Tumores Cerebrales
Sistema de red neuronal para clasificación y análisis de riesgo
"""

__version__ = "2.0"
__author__ = "Equipo de Neurooncología"

from .configuraciones import (
    TIPOS_TUMOR, CLASES_TUMOR, GRADOS_WHO,
    COLORES
)
from .redes_neuronales import (
    RedNeuronalClasificacionTumor,
    RedNeuronalRiesgoQuirurgico
)
from .simulador_imagen import SimuladorImagenCerebral
from .cargador_dicom import CargadorDICOM
from .analyzer import AnalizadorTumorCerebral
from .visualizacion import VisualizadorResultados
from .informes import imprimir_reporte_clinico

__all__ = [
    'TIPOS_TUMOR',
    'CLASES_TUMOR',
    'GRADOS_WHO',
    'COLORES',
    'RedNeuronalClasificacionTumor',
    'RedNeuronalRiesgoQuirurgico',
    'SimuladorImagenCerebral',
    'CargadorDICOM',
    'AnalizadorTumorCerebral',
    'VisualizadorResultados',
    'imprimir_reporte_clinico',
]
