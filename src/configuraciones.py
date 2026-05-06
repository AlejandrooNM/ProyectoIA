"""
Configuración centralizada del sistema
"""

# ────────────────────────────────────────────────────────────────
# COLORES Y ESTILOS
# ────────────────────────────────────────────────────────────────

COLORES = {
    'fondo':      '#0a0e1a',
    'panel':      '#111827',
    'acento':     '#00d4ff',
    'acento2':    '#ff6b6b',
    'verde':      '#00ff88',
    'amarillo':   '#ffd166',
    'texto':      '#e0e8ff',
    'subtexto':   '#6b7a99',
    'borde':      '#1e2d45',
}

# ────────────────────────────────────────────────────────────────
# TIPOS DE TUMORES Y PROPIEDADES
# ────────────────────────────────────────────────────────────────

TIPOS_TUMOR = {
    'Glioblastoma': {
        'color_base': 0.7,
        'irregularidad': 0.8,
        'necrosis': True,
        'edema': True,
        'tama_relativo': (0.12, 0.20)
    },
    'Meningioma': {
        'color_base': 0.6,
        'irregularidad': 0.3,
        'necrosis': False,
        'edema': False,
        'tama_relativo': (0.08, 0.15)
    },
    'Astrocitoma': {
        'color_base': 0.55,
        'irregularidad': 0.5,
        'necrosis': False,
        'edema': True,
        'tama_relativo': (0.07, 0.13)
    },
    'Oligodendroglioma': {
        'color_base': 0.5,
        'irregularidad': 0.4,
        'necrosis': False,
        'edema': False,
        'tama_relativo': (0.06, 0.12)
    },
    'Schwannoma': {
        'color_base': 0.58,
        'irregularidad': 0.25,
        'necrosis': False,
        'edema': False,
        'tama_relativo': (0.04, 0.09)
    }
}

# ────────────────────────────────────────────────────────────────
# CLASES Y GRADOS WHO
# ────────────────────────────────────────────────────────────────

CLASES_TUMOR = ['Glioblastoma', 'Meningioma', 'Astrocitoma', 
                'Oligodendroglioma', 'Schwannoma']

GRADOS_WHO = {
    'Glioblastoma': 'WHO Grado IV',
    'Meningioma': 'WHO Grado I-II',
    'Astrocitoma': 'WHO Grado II-III',
    'Oligodendroglioma': 'WHO Grado II-III',
    'Schwannoma': 'WHO Grado I'
}

# ────────────────────────────────────────────────────────────────
# PARÁMETROS DE REDES NEURONALES
# ────────────────────────────────────────────────────────────────

# Red de Clasificación
RED_CLASIFICACION = {
    'arquitectura': [10, 32, 64, 32, 5],
    'activaciones': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'softmax'],
    'epocas': 150,
    'lr': 0.001,
    'batch_size': 16,
    'datos_sinteticos_por_clase': 100,
    'seed': 42,
}

# Red de Riesgo Quirúrgico
RED_RIESGO = {
    'arquitectura': [8, 24, 12, 1],
    'activaciones': ['relu', 'relu', 'sigmoid'],
    'epocas': 120,
    'lr': 0.001,
    'datos_sinteticos': 500,
    'seed': 99,
}

# ────────────────────────────────────────────────────────────────
# PATRONES PARA DATOS SINTÉTICOS
# ────────────────────────────────────────────────────────────────

PATRONES_CLASIFICACION = {
    'Glioblastoma':      [0.75, 0.18, 0.70, 0.35, 0.30, 0.25, 0.55, 0.85, 0.30, 0.85],
    'Meningioma':        [0.62, 0.08, 0.85, 0.15, 0.20, 0.12, 0.55, 0.70, 0.15, 0.60],
    'Astrocitoma':       [0.55, 0.12, 0.78, 0.20, 0.25, 0.15, 0.45, 0.65, 0.20, 0.70],
    'Oligodendroglioma': [0.50, 0.10, 0.80, 0.18, 0.22, 0.10, 0.42, 0.60, 0.18, 0.65],
    'Schwannoma':        [0.58, 0.07, 0.88, 0.12, 0.40, 0.08, 0.52, 0.65, 0.13, 0.55],
}

DESVIACIONES_CLASIFICACION = {
    'Glioblastoma':      [0.08, 0.05, 0.06, 0.07, 0.10, 0.06, 0.08, 0.07, 0.07, 0.07],
    'Meningioma':        [0.06, 0.03, 0.05, 0.04, 0.08, 0.04, 0.06, 0.06, 0.05, 0.06],
    'Astrocitoma':       [0.07, 0.04, 0.06, 0.05, 0.09, 0.05, 0.07, 0.07, 0.06, 0.07],
    'Oligodendroglioma': [0.06, 0.03, 0.05, 0.04, 0.08, 0.04, 0.06, 0.06, 0.05, 0.06],
    'Schwannoma':        [0.06, 0.03, 0.04, 0.03, 0.09, 0.03, 0.06, 0.06, 0.05, 0.05],
}

# ────────────────────────────────────────────────────────────────
# PARÁMETROS DE SIMULACIÓN
# ────────────────────────────────────────────────────────────────

SIMULACION = {
    'tam_cerebro': 64,
    'seed_simulacion': 123,
}

# ────────────────────────────────────────────────────────────────
# VALORES POR DEFECTO
# ────────────────────────────────────────────────────────────────

VALORES_DEFECTO = {
    'edad': 58,
    'karnofsky': 80,
}

# ────────────────────────────────────────────────────────────────
# CURVAS DE SUPERVIVENCIA (Kaplan-Meier aproximadas)
# ────────────────────────────────────────────────────────────────

CURVAS_SUPERVIVENCIA = {
    'Glioblastoma': {
        'tiempo': [0, 6, 12, 18, 24, 36, 48, 60],
        'supervivencia': [1.0, 0.75, 0.50, 0.30, 0.20, 0.10, 0.05, 0.02]
    },
    'Meningioma': {
        'tiempo': [0, 12, 24, 60, 120, 180],
        'supervivencia': [1.0, 0.99, 0.97, 0.92, 0.85, 0.78]
    },
    'Astrocitoma': {
        'tiempo': [0, 12, 24, 36, 60, 84, 120],
        'supervivencia': [1.0, 0.90, 0.78, 0.65, 0.45, 0.30, 0.18]
    },
    'Oligodendroglioma': {
        'tiempo': [0, 12, 36, 60, 96, 120, 144],
        'supervivencia': [1.0, 0.95, 0.85, 0.70, 0.55, 0.40, 0.28]
    },
    'Schwannoma': {
        'tiempo': [0, 12, 60, 120, 180, 240],
        'supervivencia': [1.0, 1.0, 0.98, 0.95, 0.92, 0.88]
    },
}

# ────────────────────────────────────────────────────────────────
# PRONÓSTICOS
# ────────────────────────────────────────────────────────────────

PRONOSTICOS_BASE = {
    'Glioblastoma': ('Reservado', '12-18 meses (mediana)'),
    'Meningioma': ('Favorable', '15+ años (con resección completa)'),
    'Astrocitoma': ('Moderado', '3-7 años (Grado II-III)'),
    'Oligodendroglioma': ('Moderado-Favorable', '5-12 años'),
    'Schwannoma': ('Excelente', '>20 años'),
}
