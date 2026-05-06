"""
Funciones auxiliares del sistema
"""

import os
from pathlib import Path
from .configuraciones import VALORES_DEFECTO


def pedir_datos_paciente():
    """Solicita edad y Karnofsky al usuario con valores por defecto"""
    print()
    try:
        edad_str = input(f"  Edad del paciente (Enter = {VALORES_DEFECTO['edad']}): ").strip()
        edad = int(edad_str) if edad_str else VALORES_DEFECTO['edad']
    except ValueError:
        edad = VALORES_DEFECTO['edad']

    try:
        kar_str = input(f"  Escala Karnofsky 0-100 (Enter = {VALORES_DEFECTO['karnofsky']}): ").strip()
        karnofsky = int(kar_str) if kar_str else VALORES_DEFECTO['karnofsky']
        karnofsky = max(0, min(100, karnofsky))
    except ValueError:
        karnofsky = VALORES_DEFECTO['karnofsky']

    return edad, karnofsky


def validar_ruta_dicom(ruta_str):
    """Valida que la ruta DICOM existe"""
    ruta = Path(ruta_str.strip().strip('"').strip("'"))
    return ruta if ruta.exists() else None


def buscar_archivos_dicom(ruta):
    """Busca archivos DICOM en una ruta"""
    ruta = Path(ruta)
    archivos = list(ruta.glob('*.dcm')) + list(ruta.glob('**/*.dcm'))
    return archivos


def crear_directorio_salida(nombre_base='reporte'):
    """Crea directorio para guardar reportes si no existe"""
    directorio = Path(nombre_base)
    directorio.mkdir(exist_ok=True)
    return directorio
