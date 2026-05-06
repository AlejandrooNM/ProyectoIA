"""
Simulador de imágenes cerebrales MRI con tumores sintéticos
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from .configuraciones import TIPOS_TUMOR, SIMULACION


class SimuladorImagenCerebral:
    """Genera imágenes sintéticas realistas de MRI cerebral"""

    @staticmethod
    def generar_cerebro_base(tam=64):
        """Genera imagen base del cerebro con estructura anatómica"""
        img = np.zeros((tam, tam))
        cx, cy = tam // 2, tam // 2

        # Cráneo
        for i in range(tam):
            for j in range(tam):
                dx = (i - cx) / (cx * 1.0)
                dy = (j - cy) / (cy * 0.85)
                r = np.sqrt(dx**2 + dy**2)
                if 0.82 < r < 0.95:
                    img[i, j] = 0.9 + np.random.normal(0, 0.02)
                elif r < 0.82:
                    img[i, j] = 0.3 + 0.15 * np.sin(i * 0.3) * np.cos(j * 0.3)
                    img[i, j] += np.random.normal(0, 0.03)

        # Ventrículos (oscuros)
        for i in range(tam):
            for j in range(tam):
                dx = abs(i - cx) - 3
                dy = abs(j - cy) - 5
                if max(dx, 0)**2 + max(dy, 0)**2 < 25:
                    img[i, j] = 0.05 + np.random.normal(0, 0.01)

        # Surcos cerebrales
        for angulo in np.linspace(0, 2*np.pi, 20):
            r_sulco = (cx * 0.5) * (1 + 0.3 * np.sin(3 * angulo))
            xi = int(cx + r_sulco * np.cos(angulo))
            yi = int(cy + r_sulco * np.sin(angulo))
            if 0 < xi < tam and 0 < yi < tam:
                img[max(0,xi-1):xi+2, max(0,yi-1):yi+2] *= 0.7

        # Suavizado gaussiano
        img = gaussian_filter(img, sigma=0.8)
        return np.clip(img, 0, 1)

    @staticmethod
    def agregar_tumor(img_base, tipo_tumor='Glioblastoma', posicion=None):
        """Agrega tumor con características morfológicas del tipo especificado"""
        tam = img_base.shape[0]
        img = img_base.copy()
        props = TIPOS_TUMOR[tipo_tumor]

        cx, cy = tam // 2, tam // 2

        # Posición aleatoria
        if posicion is None:
            angulo = np.random.uniform(0, 2 * np.pi)
            radio = np.random.uniform(0.1, 0.5) * cx * 0.7
            tx = int(cx + radio * np.cos(angulo))
            ty = int(cy + radio * np.sin(angulo))
        else:
            tx, ty = posicion

        # Tamaño del tumor
        tmin, tmax = props['tama_relativo']
        radio_tumor = np.random.uniform(tmin, tmax) * tam

        # Forma irregular
        for i in range(tam):
            for j in range(tam):
                dx = i - tx
                dy = j - ty

                angulo_p = np.arctan2(dy, dx)
                ruido = props['irregularidad'] * np.random.normal(0, 0.15)
                radio_local = radio_tumor * (1 + ruido * np.sin(3 * angulo_p))

                dist = np.sqrt(dx**2 + dy**2)

                if dist < radio_local:
                    factor = 1 - (dist / radio_local)

                    if props['necrosis'] and dist < radio_local * 0.4:
                        img[i, j] = 0.08 + np.random.normal(0, 0.02)
                    else:
                        intensidad = props['color_base'] + 0.2 * factor
                        img[i, j] = np.clip(
                            intensidad + np.random.normal(0, 0.03), 0, 1
                        )

                elif dist < radio_local * 1.4 and props['edema']:
                    factor_edema = 1 - (dist - radio_local) / (radio_local * 0.4)
                    img[i, j] = np.clip(
                        img[i, j] * 0.8 + 0.1 * factor_edema, 0, 1
                    )

        # Suavizado
        img = gaussian_filter(img, sigma=0.5)
        return np.clip(img, 0, 1), (tx, ty, radio_tumor)

    @staticmethod
    def extraer_features(img, posicion_tumor):
        """Extrae 10 características numéricas de la imagen"""
        tx, ty, radio = posicion_tumor
        tam = img.shape[0]
        cx, cy = tam // 2, tam // 2

        # Crear máscara del tumor
        mask = np.zeros_like(img)
        for i in range(tam):
            for j in range(tam):
                if (i-tx)**2 + (j-ty)**2 < radio**2:
                    mask[i, j] = 1

        # 1. Intensidad media
        intensidad_media = np.mean(img[mask > 0]) if mask.sum() > 0 else 0

        # 2. Heterogeneidad
        heterogeneidad = np.std(img[mask > 0]) if mask.sum() > 0 else 0

        # 3. Simetría
        img_flip = np.fliplr(img)
        simetria = 1 - np.mean(np.abs(img - img_flip))

        # 4. Gradiente en bordes
        gx = np.gradient(img, axis=0)
        gy = np.gradient(img, axis=1)
        magnitud_grad = np.sqrt(gx**2 + gy**2)
        gradiente_borde = np.mean(magnitud_grad[mask > 0]) if mask.sum() > 0 else 0

        # 5. Localización relativa
        distancia_centro = np.sqrt((tx - cx)**2 + (ty - cy)**2) / tam

        # 6. Volumen relativo
        volumen_relativo = np.pi * radio**2 / tam**2

        # 7-8. Cuantiles
        pixeles_tumor = img[mask > 0] if mask.sum() > 0 else np.array([0])
        q25 = np.percentile(pixeles_tumor, 25)
        q75 = np.percentile(pixeles_tumor, 75)
        iqr = q75 - q25

        # 9. Entropía
        hist, _ = np.histogram(pixeles_tumor, bins=10, range=(0, 1))
        hist_norm = hist / (hist.sum() + 1e-10)
        entropia = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))

        features = np.array([
            intensidad_media,
            heterogeneidad,
            simetria,
            gradiente_borde * 10,
            distancia_centro,
            volumen_relativo * 10,
            q25,
            q75,
            iqr,
            entropia / 3.32
        ])

        return features
