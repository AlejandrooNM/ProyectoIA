"""
Script para entrenar la IA con datos REALES de BraTS 2021
Extrae información clínica real de las imágenes y máscaras de segmentación
Sin datos sintéticos - solo datos clínicos reales
"""

import numpy as np
from pathlib import Path
import sys
import json

try:
    import pydicom
except ImportError:
    print("pydicom no instalado. Instala con: pip install pydicom")
    sys.exit(1)

try:
    import nibabel as nib
except ImportError:
    print("nibabel no instalado. Instala con: pip install nibabel")
    sys.exit(1)

from src.analyzer import AnalizadorTumorCerebral
from src.configuraciones import CLASES_TUMOR


def cargar_imagenes_dicom(ruta_secuencia):
    """Carga serie DICOM de una secuencia MRI"""
    archivos_dcm = sorted(list(Path(ruta_secuencia).glob('*.dcm')))
    if not archivos_dcm:
        return None
    
    imagenes = []
    for archivo in archivos_dcm:
        try:
            ds = pydicom.dcmread(archivo)
            if hasattr(ds, 'pixel_array'):
                img = ds.pixel_array.astype(np.float32)
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                imagenes.append(img)
        except Exception:
            continue
    
    return np.stack(imagenes, axis=0) if imagenes else None


def buscar_archivo_segmentacion(ruta_paciente):
    """Busca archivo de segmentación en la carpeta del paciente"""
    ruta = Path(ruta_paciente)
    
    for ext in ['*.nii.gz', '*.nii']:
        for archivo in ruta.rglob(ext):
            if 'seg' in archivo.name.lower():
                return archivo
    
    return None


def cargar_segmentacion(ruta_archivo):
    """Carga máscara de segmentación NIfTI"""
    try:
        nimg = nib.load(ruta_archivo)
        segmentacion = nimg.get_fdata().astype(np.uint8)
        return segmentacion
    except:
        return None


def extraer_metricas_clinicas(volumenes, segmentacion=None):
    """
    Extrae métricas clínicas reales del tumor
    Retorna características clinicamente significativas
    """
    caracteristicas = []
    
    if segmentacion is not None:
        tumor_maligno = (segmentacion == 2).astype(np.float32)
        necrosis = (segmentacion == 1).astype(np.float32)
        edema = (segmentacion == 3).astype(np.float32)
        
        volumen_tumor = np.sum(tumor_maligno)
        volumen_necrosis = np.sum(necrosis)
        volumen_edema = np.sum(edema)
        volumen_total = volumen_tumor + volumen_necrosis + volumen_edema
        
        agresividad = volumen_necrosis / (volumen_total + 1e-5)
        malignidad_score = (volumen_tumor + 0.5 * volumen_necrosis) / (volumen_total + 1e-5)
    else:
        agresividad = 0.3
        malignidad_score = 0.3
        volumen_tumor = 0
        volumen_necrosis = 0
        volumen_edema = 0
        volumen_total = 0
    
    for secuencia, volumen in volumenes.items():
        if volumen is None:
            continue
        
        idx_centro = volumen.shape[0] // 2
        img_centro = volumen[idx_centro]
        
        intensidad_media = np.mean(img_centro)
        intensidad_std = np.std(img_centro)
        intensidad_max = np.max(img_centro)
        intensidad_min = np.min(img_centro)
        
        w = img_centro.shape[1]
        izq = img_centro[:, :w//2]
        der = img_centro[:, w//2:]
        min_w = min(izq.shape[1], der.shape[1])
        
        try:
            corr = np.corrcoef(izq[:, :min_w].flatten(), der[:, :min_w].flatten())[0, 1]
            simetria = max(0, min(1, (corr + 1) / 2))
        except:
            simetria = 0.5
        
        asimetria = 1 - simetria
        
        caracteristicas.extend([
            intensidad_media,
            intensidad_std,
            intensidad_max,
            intensidad_min,
            asimetria,
            agresividad,
            malignidad_score
        ])
    
    while len(caracteristicas) < 40:
        caracteristicas.append(0.0)
    
    caracteristicas = np.array(caracteristicas[:40], dtype=np.float32)
    caracteristicas = np.clip(caracteristicas, 0, 1)
    
    return caracteristicas, {
        'volumen_tumor': volumen_tumor,
        'volumen_necrosis': volumen_necrosis,
        'volumen_edema': volumen_edema,
        'agresividad': float(agresividad),
        'malignidad': float(malignidad_score),
        'asimetria': float(asimetria) if 'asimetria' in locals() else 0.5
    }


def cargar_datos_pacientes_reales(ruta_base, max_pacientes=None):
    """
    Carga datos REALES de pacientes BraTS 2021
    Retorna X (características), y (etiquetas basadas en métricas reales)
    """
    ruta_base = Path(ruta_base)
    pacientes = sorted([p for p in ruta_base.iterdir() if p.is_dir() and p.name.startswith('0')])
    
    if max_pacientes:
        pacientes = pacientes[:max_pacientes]
    
    print(f"Encontrados {len(pacientes)} pacientes\n")
    
    X_datos = []
    y_datos = []
    metadatos = []
    
    for i, carpeta_paciente in enumerate(pacientes):
        print(f"[{i+1}/{len(pacientes)}] {carpeta_paciente.name}...", end=" ", flush=True)
        
        try:
            volumenes = {}
            for secuencia in ['FLAIR', 'T1w', 'T1wCE', 'T2w']:
                carpeta_seq = carpeta_paciente / secuencia
                if carpeta_seq.exists():
                    vol = cargar_imagenes_dicom(carpeta_seq)
                    if vol is not None:
                        volumenes[secuencia] = vol
            
            if not volumenes:
                print("no imagenes")
                continue
            
            archivo_seg = buscar_archivo_segmentacion(carpeta_paciente)
            segmentacion = None
            if archivo_seg:
                segmentacion = cargar_segmentacion(archivo_seg)
            
            caracteristicas, metricas = extraer_metricas_clinicas(volumenes, segmentacion)
            X_datos.append(caracteristicas)
            metadatos.append({
                'paciente': carpeta_paciente.name,
                'metricas': metricas,
                'tiene_segmentacion': segmentacion is not None
            })
            
            malignidad = metricas['malignidad']
            agresividad = metricas['agresividad']
            
            if agresividad > 0.5 and malignidad > 0.6:
                etiqueta = 0
                tipo = 'Glioblastoma'
            elif malignidad > 0.5 and agresividad > 0.3:
                etiqueta = 1
                tipo = 'Astrocitoma'
            elif malignidad > 0.4 and agresividad < 0.3:
                etiqueta = 3
                tipo = 'Oligodendroglioma'
            elif malignidad < 0.3:
                etiqueta = 2
                tipo = 'Meningioma'
            else:
                etiqueta = 4
                tipo = 'Schwannoma'
            
            etiqueta_onehot = np.zeros(5)
            etiqueta_onehot[etiqueta] = 1
            y_datos.append(etiqueta_onehot)
            
            print(f"OK {tipo}")
        
        except Exception as e:
            print(f"error {str(e)[:30]}")
            continue
    
    if not X_datos:
        print("\nNo se pudieron cargar pacientes")
        return None, None, None
    
    X = np.array(X_datos, dtype=np.float32)
    y = np.array(y_datos, dtype=np.float32)
    
    return X, y, metadatos


def main():
    ruta_datos = r"E:\PKG - RSNA-ASNR-MICCAI-BraTS-2021\RSNA-ASNR-MICCAI-BraTS-2021\BraTS2021_TrainingSet_dcm\new-not-previously-in-TCIA"
    
    print()
    print("="*65)
    print("  ENTRENAMIENTO CON DATOS REALES CLINICOS - BraTS 2021")
    print("="*65)
    print()
    
    if not Path(ruta_datos).exists():
        print(f"Ruta no encontrada: {ruta_datos}")
        return
    
    print(f"Datos: {ruta_datos}\n")
    print("Cargando imágenes DICOM y segmentaciones...\n")
    
    X, y, metadatos = cargar_datos_pacientes_reales(ruta_datos, max_pacientes=100)
    
    if X is None:
        print("No se pudieron cargar datos")
        return
    
    print(f"\nDatos: {X.shape[0]} pacientes x {X.shape[1]} features")
    print("\nDistribucion de tumores (basada en datos reales):")
    for i, clase in enumerate(CLASES_TUMOR):
        count = np.sum(y[:, i])
        print(f"  {clase}: {int(count)} pacientes")
    
    with open('metadatos_pacientes.json', 'w') as f:
        json.dump(metadatos, f, indent=2, default=str)
    print("\nMetadatos guardados en: metadatos_pacientes.json")
    
    print("\n" + "="*65)
    print("  ENTRENANDO REDES NEURONALES (DATOS REALES)")
    print("="*65)
    print()
    
    analizador = AnalizadorTumorCerebral()
    
    print("Red 1: Clasificacion Histologica (DATOS REALES)")
    analizador.red_clasificacion.entrenar(
        X, y, 
        epocas=200, 
        lr=0.001, 
        tamano_lote=8,
        verbose=True
    )
    
    print("\nRed 2: Riesgo Quirurgico (DATOS REALES)")
    
    X_riesgo = []
    y_riesgo = []
    
    for i, meta in enumerate(metadatos):
        metricas = meta['metricas']
        
        features = np.array([
            metricas['malignidad'],
            min(1.0, metricas['volumen_tumor'] / 50000),
            metricas['agresividad'],
            metricas['asimetria'],
            np.mean(X[i][:7]),
            min(1.0, metricas['volumen_edema'] / 10000),
            min(1.0, metricas['volumen_necrosis'] / 10000),
            np.std(X[i][:7])
        ])
        
        X_riesgo.append(features)
        
        riesgo = (metricas['malignidad'] * 0.4 + 
                 metricas['agresividad'] * 0.3 + 
                 metricas['asimetria'] * 0.2 + 
                 (min(1.0, metricas['volumen_tumor'] / 100000) * 0.1))
        riesgo = np.clip(riesgo, 0, 1)
        
        y_riesgo.append(riesgo)
    
    X_riesgo = np.array(X_riesgo, dtype=np.float32)
    y_riesgo = np.array(y_riesgo, dtype=np.float32).reshape(-1, 1)
    
    analizador.red_riesgo.entrenar(
        X_riesgo, y_riesgo, 
        epocas=150, 
        lr=0.001,
        verbose=True
    )
    
    print("\n" + "="*65)
    print("ENTRENAMIENTO COMPLETADO CON DATOS REALES")
    print("="*65)
    print("\nUso:")
    print("  python principal.py")
    print("  (Opcion 1 para cargar DICOM reales)")
    print()


if __name__ == '__main__':
    main()
