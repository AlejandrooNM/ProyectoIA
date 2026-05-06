"""
Analizador principal que integra las redes neuronales
"""

import numpy as np
import time
from .redes_neuronales import (
    RedNeuronalClasificacionTumor, 
    RedNeuronalRiesgoQuirurgico
)
from .simulador_imagen import SimuladorImagenCerebral
from .configuraciones import PRONOSTICOS_BASE, TIPOS_TUMOR


class AnalizadorTumorCerebral:
    """Sistema integrado de análisis de tumor cerebral"""

    def __init__(self):
        self.red_clasificacion = RedNeuronalClasificacionTumor()
        self.red_riesgo = RedNeuronalRiesgoQuirurgico()
        self.simulador = SimuladorImagenCerebral()
        self.sistemas_entrenados = False
        self.resultado = None

    def inicializar_y_entrenar(self):
        """Entrena ambas redes neuronales con datos sintéticos"""
        print("=" * 65)
        print("  INICIALIZANDO SISTEMA DE ANÁLISIS NEURAL")
        print("=" * 65)
        print()

        # Entrenar red de clasificación
        print("► Red Neuronal 1: Clasificación Histológica")
        X_cls, y_cls = self.red_clasificacion.generar_datos_sinteticos(n_por_clase=100)
        self.red_clasificacion.entrenar(X_cls, y_cls, epocas=150, lr=0.001)

        # Entrenar red de riesgo
        print("► Red Neuronal 2: Score de Riesgo Quirúrgico")
        X_rsg, y_rsg = self.red_riesgo.generar_datos_riesgo(n=500)
        self.red_riesgo.entrenar(X_rsg, y_rsg, epocas=120, lr=0.001)

        self.sistemas_entrenados = True
        print("=" * 65)
        print("  ✓ Sistema listo para análisis")
        print("=" * 65)
        print()

    def analizar(self, tipo_real=None, edad_paciente=65, karnofsky=80):
        """Realiza análisis completo de una imagen cerebral simulada"""
        if not self.sistemas_entrenados:
            self.inicializar_y_entrenar()

        # Seleccionar tipo de tumor
        tipos = list(TIPOS_TUMOR.keys())
        if tipo_real is None:
            tipo_real = np.random.choice(tipos)

        print(f"\n  Procesando imagen de paciente...")
        print(f"  Edad: {edad_paciente} años │ Karnofsky: {karnofsky}")
        time.sleep(0.3)

        # Generar imagen
        img_base = SimuladorImagenCerebral.generar_cerebro_base(tam=64)
        img_tumor, pos_tumor = SimuladorImagenCerebral.agregar_tumor(
            img_base, tipo_tumor=tipo_real
        )
        tx, ty, radio = pos_tumor

        # Extraer características
        features = SimuladorImagenCerebral.extraer_features(img_tumor, pos_tumor)

        # ── Predicción de clasificación ──
        probs_tipo = self.red_clasificacion.predecir(features)
        idx_predicho = np.argmax(probs_tipo)
        tipo_predicho = self.red_clasificacion.CLASES[idx_predicho]
        confianza = probs_tipo[idx_predicho]
        grado_who = self.red_clasificacion.GRADOS[tipo_predicho]

        # ── Predicción de riesgo quirúrgico ──
        es_maligno = 1.0 if tipo_predicho == 'Glioblastoma' else (
            0.6 if tipo_predicho in ['Astrocitoma', 'Oligodendroglioma'] else 0.2
        )
        tam_64 = 64
        cx, cy = tam_64 // 2, tam_64 // 2
        profundidad = min(1.0, np.sqrt((tx - cx)**2 + (ty - cy)**2) / (tam_64 * 0.4))
        area_elocuente = max(0, 1 - profundidad) * 0.8

        features_riesgo = np.array([
            es_maligno,
            np.pi * radio**2 / (tam_64**2),
            profundidad,
            area_elocuente,
            np.random.uniform(0.2, 0.7),
            min(1.0, max(0, (edad_paciente - 18) / 82)),
            (100 - karnofsky) / 100,
            float(features[4])
        ])
        score_riesgo = self.red_riesgo.predecir(features_riesgo)

        # ── Cálculo de métricas clínicas ──
        volumen_mm3 = (4/3) * np.pi * (radio * 3.0)**3
        diametro_mm = 2 * radio * 3.0

        pronostico, supervivencia = self._calcular_pronostico(
            tipo_predicho, score_riesgo, karnofsky, edad_paciente
        )
        recomendaciones = self._generar_recomendaciones(
            tipo_predicho, score_riesgo, diametro_mm, grado_who
        )

        self.resultado = {
            'tipo_real': tipo_real,
            'tipo_predicho': tipo_predicho,
            'confianza': confianza,
            'probabilidades': dict(zip(self.red_clasificacion.CLASES, probs_tipo)),
            'grado_who': grado_who,
            'score_riesgo': score_riesgo,
            'diametro_mm': diametro_mm,
            'volumen_mm3': volumen_mm3,
            'posicion_tumor': pos_tumor,
            'pronostico': pronostico,
            'supervivencia_estimada': supervivencia,
            'recomendaciones': recomendaciones,
            'img_base': img_base,
            'img_tumor': img_tumor,
            'features': features,
            'edad_paciente': edad_paciente,
            'karnofsky': karnofsky,
        }

        return self.resultado

    def _calcular_pronostico(self, tipo, score_riesgo, karnofsky, edad):
        """Estima pronóstico según tipo tumoral y factores clínicos"""
        prog_base, sv = PRONOSTICOS_BASE.get(tipo, ('Indeterminado', 'N/A'))
        if karnofsky < 70 or edad > 70:
            prog_base = prog_base + ' (factores de mal pronóstico)'
        return prog_base, sv

    def _generar_recomendaciones(self, tipo, score_riesgo, diametro, grado):
        """Genera recomendaciones clínicas"""
        recs = []

        if score_riesgo > 70:
            recs.append("URGENTE: Evaluación multidisciplinaria inmediata")
            recs.append("Considerar corticoides (Dexametasona) para edema")
        elif score_riesgo > 40:
            recs.append("Programar cirugía en < 2 semanas")
            recs.append("Evaluación neuropsicológica prequirúrgica")

        if tipo == 'Glioblastoma':
            recs.append("Protocolo Stupp: RT + Temozolomida concurrente + adyuvante")
            recs.append("Solicitar análisis MGMT metilación y mutación IDH")
            recs.append("Considerar bevacizumab en recidiva")
        elif tipo in ['Astrocitoma', 'Oligodendroglioma']:
            recs.append("Resección máxima segura + RT adyuvante")
            recs.append("Análisis de mutación IDH1/IDH2 y codeleción 1p/19q")
        elif tipo == 'Meningioma':
            recs.append("Resección completa preferible (Simpson Grado I)")
            recs.append("Seguimiento con RM cada 6-12 meses post-cirugía")
        elif tipo == 'Schwannoma':
            recs.append("Radiocirugía estereotáctica vs. microcirugía")
            recs.append("Preservación del nervio facial/auditivo prioritaria")

        if diametro > 40:
            recs.append(f"Tumor de gran tamaño ({diametro:.1f}mm): planificar acceso quirúrgico")
        if diametro < 15:
            recs.append(f"Tumor pequeño ({diametro:.1f}mm): considerar seguimiento con RM")

        recs.append("RM funcional y tractografía DTI prequirúrgica")
        recs.append("Monitoreo neurofisiológico intraoperatorio recomendado")

        return recs
