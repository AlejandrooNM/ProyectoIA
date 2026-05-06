"""
Cargador de imágenes DICOM reales
"""

import numpy as np
from pathlib import Path

# Intentar importar librerías opcionales
try:
    import pydicom
    PYDICOM_DISPONIBLE = True
except ImportError:
    PYDICOM_DISPONIBLE = False

try:
    from skimage import filters, morphology, exposure
    from skimage.filters import threshold_otsu
    SKIMAGE_DISPONIBLE = True
except ImportError:
    SKIMAGE_DISPONIBLE = False

try:
    import cv2
    CV2_DISPONIBLE = True
except ImportError:
    CV2_DISPONIBLE = False


class CargadorDICOM:
    """Carga y preprocesa imágenes DICOM reales de tumores cerebrales"""

    @staticmethod
    def cargar_serie(carpeta):
        """Carga una serie DICOM desde una carpeta"""
        if not PYDICOM_DISPONIBLE:
            print("  ⚠  pydicom no instalado. Instálalo con: pip install pydicom")
            return None, None

        carpeta = Path(carpeta)
        archivos = sorted(carpeta.glob('*.dcm'))
        if not archivos:
            archivos = sorted(carpeta.rglob('*.dcm'))

        if not archivos:
            print(f"  ⚠  No se encontraron archivos .dcm en: {carpeta}")
            return None, None

        slices = []
        for f in archivos:
            try:
                ds = pydicom.dcmread(str(f))
                if hasattr(ds, 'PixelData'):
                    slices.append(ds)
            except Exception:
                continue

        if not slices:
            return None, None

        # Ordenar por posición Z
        def pos_z(s):
            if hasattr(s, 'ImagePositionPatient'):
                try:
                    return float(s.ImagePositionPatient[2])
                except Exception:
                    pass
            if hasattr(s, 'SliceLocation'):
                try:
                    return float(s.SliceLocation)
                except Exception:
                    pass
            return float(getattr(s, 'InstanceNumber', 0))

        slices.sort(key=pos_z)

        # Construir volumen 3D
        forma_ref = None
        pixeles = []
        for s in slices:
            try:
                pix = s.pixel_array.astype(np.float32)
                if forma_ref is None:
                    forma_ref = pix.shape
                if pix.shape == forma_ref:
                    pixeles.append(pix)
            except Exception:
                continue

        if not pixeles:
            return None, None

        volumen = np.stack(pixeles, axis=0)

        # Espaciado
        primera = slices[0]
        try:
            ps = primera.PixelSpacing
            px, py = float(ps[0]), float(ps[1])
        except Exception:
            px = py = 1.0

        st = 1.0
        if hasattr(primera, 'SliceThickness'):
            try:
                st = float(primera.SliceThickness)
            except Exception:
                st = 1.0

        return volumen, (px, py, st)

    @staticmethod
    def normalizar(volumen):
        """Normaliza el volumen a rango [0, 1]"""
        vmin, vmax = np.min(volumen), np.max(volumen)
        if vmax > vmin:
            return (volumen - vmin) / (vmax - vmin)
        return np.zeros_like(volumen)

    @staticmethod
    def segmentar_tumor(volumen):
        """Segmenta el tumor con Otsu o umbral simple"""
        mascara = np.zeros_like(volumen, dtype=bool)
        for i in range(volumen.shape[0]):
            slc = volumen[i]
            try:
                if SKIMAGE_DISPONIBLE:
                    thresh = threshold_otsu(slc)
                else:
                    thresh = np.mean(slc) + np.std(slc)
                mascara[i] = slc > thresh
            except Exception:
                mascara[i] = slc > np.mean(slc)

        # Limpiar pequeños objetos
        if SKIMAGE_DISPONIBLE:
            try:
                mascara = morphology.remove_small_objects(mascara, min_size=50)
                mascara = morphology.binary_closing(mascara, morphology.ball(2))
            except Exception:
                pass

        # Quedarse con la componente más grande
        try:
            from scipy.ndimage import label
            labeled, n = label(mascara)
            if n > 0:
                sizes = np.bincount(labeled.ravel())
                if len(sizes) > 1:
                    mayor = np.argmax(sizes[1:]) + 1
                    mascara = labeled == mayor
        except Exception:
            pass

        return mascara

    @staticmethod
    def calcular_metricas(mascara, espaciado):
        """Calcula métricas del tumor"""
        px, py, st = espaciado
        voxels = int(mascara.sum())
        if voxels == 0:
            return {'volumen_mm3': 0, 'volumen_cm3': 0, 'diametro_mm': 0}

        vol_mm3 = voxels * px * py * st
        vol_cm3 = vol_mm3 / 1000.0
        diametro_mm = 2 * ((3 * vol_cm3 / (4 * np.pi)) ** (1/3)) * 10

        return {
            'volumen_mm3': float(vol_mm3),
            'volumen_cm3': float(vol_cm3),
            'diametro_mm': float(diametro_mm),
            'voxels': voxels,
            'n_slices': int(np.sum(mascara.sum(axis=(1, 2)) > 0)),
        }

    @staticmethod
    def extraer_slice_central(volumen, mascara):
        """Extrae el slice central con mayor área de tumor"""
        areas = mascara.sum(axis=(1, 2))
        idx = int(np.argmax(areas)) if areas.max() > 0 else volumen.shape[0] // 2
        return volumen[idx], mascara[idx], idx

    @staticmethod
    def extraer_features_dicom(volumen, mascara):
        """Extrae 10 features para la red neuronal"""
        vox = volumen[mascara] if mascara.sum() > 0 else volumen.ravel()

        intensidad_media = float(np.mean(vox))
        heterogeneidad = float(np.std(vox))

        # Simetría
        mid = volumen.shape[2] // 2
        izq = volumen[:, :, :mid]
        der = np.flip(volumen[:, :, mid:], axis=2)
        min_w = min(izq.shape[2], der.shape[2])
        simetria = float(1 - np.mean(np.abs(izq[:, :, :min_w] - der[:, :, :min_w])))

        # Gradiente
        gz, gy, gx = np.gradient(volumen)
        grad_mag = np.sqrt(gz**2 + gy**2 + gx**2)
        gradiente_borde = float(np.mean(grad_mag[mascara]) * 10) if mascara.sum() > 0 else 0.0

        # Localización
        if mascara.sum() > 0:
            coords = np.array(np.where(mascara)).mean(axis=1)
            centro = np.array(volumen.shape) / 2
            distancia_centro = float(np.linalg.norm(coords - centro) / np.linalg.norm(centro))
        else:
            distancia_centro = 0.0

        volumen_relativo = float(mascara.sum() / volumen.size * 10)

        q25 = float(np.percentile(vox, 25))
        q75 = float(np.percentile(vox, 75))
        iqr = float(q75 - q25)

        hist, _ = np.histogram(vox, bins=10, range=(0, 1))
        h = hist / (hist.sum() + 1e-10)
        entropia = float(-np.sum(h * np.log2(h + 1e-10)) / 3.32)

        return np.clip(np.array([
            intensidad_media, heterogeneidad, simetria,
            gradiente_borde, distancia_centro, volumen_relativo,
            q25, q75, iqr, entropia
        ]), 0, 1)
