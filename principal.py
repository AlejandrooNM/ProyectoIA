"""
Programa principal - Análisis de Tumores Cerebrales
Sistema de red neuronal para clasificación histológica y evaluación de riesgo quirúrgico
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from datetime import datetime
from scipy.ndimage import zoom

from src.analyzer import AnalizadorTumorCerebral
from src.simulador_imagen import SimuladorImagenCerebral
from src.cargador_dicom import CargadorDICOM
from src.visualizacion import VisualizadorResultados
from src.informes import imprimir_reporte_clinico
from src.utilidades import (
    pedir_datos_paciente, 
    validar_ruta_dicom, 
    buscar_archivos_dicom
)


def modo_simulado(analizador, edad_paciente, karnofsky):
    """Ejecuta el análisis en modo simulado (sin DICOM)"""
    np.random.seed(int(time.time()) % 10000)
    tipos_disponibles = list(SimuladorImagenCerebral.TIPOS_TUMOR.keys())
    tipo_demo = tipos_disponibles[np.random.randint(len(tipos_disponibles))]

    print(f"\n  Tipo de tumor (demo aleatorio): {tipo_demo}")

    resultado = analizador.analizar(
        tipo_real=tipo_demo,
        edad_paciente=edad_paciente,
        karnofsky=karnofsky
    )
    return resultado


def modo_dicom(ruta_dicom, analizador, edad_paciente, karnofsky):
    """
    Carga imágenes DICOM reales, extrae features y ejecuta
    las redes neuronales sobre ellas
    """
    print(f"\n  Cargando imágenes DICOM desde: {ruta_dicom}")

    ruta = Path(ruta_dicom)
    archivos_dcm = list(ruta.glob('*.dcm')) + list(ruta.glob('**/*.dcm'))

    if not archivos_dcm:
        print("  ⚠  No se encontraron archivos .dcm — cambiando a modo simulado.")
        return modo_simulado(analizador, edad_paciente, karnofsky)

    # Intentar cargar la serie
    volumen, espaciado = CargadorDICOM.cargar_serie(ruta)
    if volumen is None:
        print("  ⚠  No se pudo cargar la serie DICOM — cambiando a modo simulado.")
        return modo_simulado(analizador, edad_paciente, karnofsky)

    print(f"  ✓ Serie cargada: {volumen.shape[0]} slices  │  "
          f"Resolución: {volumen.shape[1]}×{volumen.shape[2]} px  │  "
          f"Espaciado: {espaciado[0]:.2f}×{espaciado[1]:.2f}×{espaciado[2]:.2f} mm")

    # Normalizar y segmentar
    volumen_norm = CargadorDICOM.normalizar(volumen)
    print("  Segmentando región tumoral...")
    mascara = CargadorDICOM.segmentar_tumor(volumen_norm)
    metricas_dicom = CargadorDICOM.calcular_metricas(mascara, espaciado)

    print(f"  ✓ Segmentación completada  │  "
          f"Voxels tumorales: {metricas_dicom['voxels']:,}  │  "
          f"Volumen: {metricas_dicom['volumen_cm3']:.2f} cm³")

    # Extraer slice central
    slice_img, slice_mask, idx_central = CargadorDICOM.extraer_slice_central(
        volumen_norm, mascara
    )

    # Extraer features
    features = CargadorDICOM.extraer_features_dicom(volumen_norm, mascara)

    # ── Clasificación ──
    probs_tipo = analizador.red_clasificacion.predecir(features)
    idx_pred = int(np.argmax(probs_tipo))
    tipo_predicho = analizador.red_clasificacion.CLASES[idx_pred]
    confianza = float(probs_tipo[idx_pred])
    grado_who = analizador.red_clasificacion.GRADOS[tipo_predicho]

    # ── Score de riesgo ──
    es_maligno = 1.0 if tipo_predicho == 'Glioblastoma' else (
        0.6 if tipo_predicho in ['Astrocitoma', 'Oligodendroglioma'] else 0.2
    )

    features_riesgo = np.array([
        es_maligno,
        min(1.0, metricas_dicom['volumen_cm3'] / 100.0),
        float(features[4]),
        max(0, 1 - float(features[4])) * 0.8,
        np.random.uniform(0.2, 0.7),
        min(1.0, max(0, (edad_paciente - 18) / 82)),
        (100 - karnofsky) / 100,
        float(features[3]) / 10,
    ])
    score_riesgo = analizador.red_riesgo.predecir(features_riesgo)

    # ── Calcular pronóstico y recomendaciones ──
    pronostico, supervivencia = analizador._calcular_pronostico(
        tipo_predicho, score_riesgo, karnofsky, edad_paciente
    )
    recomendaciones = analizador._generar_recomendaciones(
        tipo_predicho, score_riesgo, metricas_dicom['diametro_mm'], grado_who
    )

    # ── Redimensionar imagen ──
    factor = 64 / slice_img.shape[0]
    img_64 = zoom(slice_img, factor, order=1)
    img_64 = np.clip(img_64, 0, 1)

    if slice_mask.sum() > 0:
        coords = np.array(np.where(slice_mask)).mean(axis=1) * factor
        tx, ty = int(coords[0]), int(coords[1])
        area_px = slice_mask.sum()
        radio_2d = np.sqrt(area_px / np.pi) * factor
    else:
        tx, ty = 32, 32
        radio_2d = 8.0

    resultado = {
        'tipo_real': f'DICOM ({ruta.name})',
        'tipo_predicho': tipo_predicho,
        'confianza': confianza,
        'probabilidades': dict(zip(analizador.red_clasificacion.CLASES, probs_tipo)),
        'grado_who': grado_who,
        'score_riesgo': score_riesgo,
        'diametro_mm': metricas_dicom['diametro_mm'],
        'volumen_mm3': metricas_dicom['volumen_mm3'],
        'posicion_tumor': (tx, ty, radio_2d),
        'pronostico': pronostico,
        'supervivencia_estimada': supervivencia,
        'recomendaciones': recomendaciones,
        'img_base': img_64,
        'img_tumor': img_64,
        'features': features,
        'edad_paciente': edad_paciente,
        'karnofsky': karnofsky,
    }
    return resultado


def main():
    """Función principal"""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   ANÁLISIS NEURAL DE TUMORES CEREBRALES  ·  v2.0  Modular  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  ¿Qué desea hacer?")
    print("  1. Cargar imágenes DICOM reales (carpeta con archivos .dcm)")
    print("  2. Ejecutar demostración con imágenes sintéticas (sin DICOM)")
    print()

    opcion = input("  Seleccione opción (1 o 2): ").strip()

    usar_dicom = opcion == "1"
    ruta_dicom = None

    if usar_dicom:
        print()
        print("  Ingrese la ruta a la carpeta con archivos DICOM.")
        print()
        ruta_dicom = input("  Ruta DICOM: ").strip()

        if not validar_ruta_dicom(ruta_dicom):
            print(f"\n  ⚠  La ruta no existe.")
            print("  Cambiando a modo simulado automáticamente.\n")
            usar_dicom = False

    # Datos del paciente
    print()
    print("  Ingrese datos del paciente (opcional, Enter para valores por defecto):")
    edad_paciente, karnofsky = pedir_datos_paciente()

    print()
    print(f"  ├─ Modo        : {'DICOM real' if usar_dicom else 'Simulado'}")
    print(f"  ├─ Edad        : {edad_paciente} años")
    print(f"  └─ Karnofsky   : {karnofsky}")
    print()

    # ── Entrenar redes neuronales ──
    analizador = AnalizadorTumorCerebral()
    analizador.inicializar_y_entrenar()

    # ── Ejecutar análisis ──
    if usar_dicom:
        resultado = modo_dicom(ruta_dicom, analizador, edad_paciente, karnofsky)
    else:
        resultado = modo_simulado(analizador, edad_paciente, karnofsky)

    # ── Reporte en consola ──
    imprimir_reporte_clinico(resultado)

    # ── Visualización ──
    print("  Generando visualización médica...")
    fig = VisualizadorResultados.generar_reporte_visual(
        resultado,
        analizador.red_clasificacion.historial_perdida,
        analizador.red_riesgo.historial_perdida
    )

    nombre_archivo = f"reporte_tumor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(nombre_archivo, dpi=150, bbox_inches='tight',
                facecolor='#0a0e1a', edgecolor='none')
    print(f"  ✓ Imagen guardada como: {nombre_archivo}")

    plt.tight_layout()
    plt.show()

    return resultado


if __name__ == '__main__':
    resultado = main()
