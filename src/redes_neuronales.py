"""
Redes neuronales para clasificación y evaluación de riesgo
"""

import numpy as np
from .capas import CapaDensa
from .configuraciones import (
    CLASES_TUMOR, RED_CLASIFICACION, RED_RIESGO,
    PATRONES_CLASIFICACION, DESVIACIONES_CLASIFICACION, GRADOS_WHO
)


class RedNeuronalClasificacionTumor:
    """
    MLP para clasificación de tipo de tumor
    Arquitectura: 10 → 32 → 64 → 32 → 5
    """

    def __init__(self):
        np.random.seed(RED_CLASIFICACION['seed'])
        self.capas = [
            CapaDensa(10, 32, 'relu_con_fugas'),
            CapaDensa(32, 64, 'relu_con_fugas'),
            CapaDensa(64, 32, 'relu_con_fugas'),
            CapaDensa(32, 5, 'softmax'),
        ]
        self.historial_perdida = []
        self.entrenado = False

    def adelante(self, x):
        """Propagación hacia adelante"""
        for capa in self.capas:
            x = capa.adelante(x)
        return x

    def calcular_perdida(self, y_pred, y_real):
        """Entropía cruzada categórica"""
        y_pred_clip = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(np.sum(y_real * np.log(y_pred_clip), axis=1))

    def atras(self, y_pred, y_real, lr=0.001):
        """Retropropagación"""
        grad = y_pred - y_real
        for capa in reversed(self.capas):
            grad = capa.atras(grad, lr)

    def entrenar(self, X, y, epocas=150, lr=0.001, tamano_lote=16, verbose=True):
        """Entrenamiento con mini-lotes"""
        n = X.shape[0]
        if verbose:
            print("  ┌─ Iniciando entrenamiento de Red de Clasificación...")

        for epoca in range(epocas):
            indices = np.random.permutation(n)
            X_barajado = X[indices]
            y_barajado = y[indices]
            perdida_epoch = 0
            n_lotes = 0

            for i in range(0, n, tamano_lote):
                X_lote = X_barajado[i:i+tamano_lote]
                y_lote = y_barajado[i:i+tamano_lote]

                y_pred = self.adelante(X_lote)
                perdida_epoch += self.calcular_perdida(y_pred, y_lote)
                self.atras(y_pred, y_lote, lr)
                n_lotes += 1

            perdida_media = perdida_epoch / n_lotes
            self.historial_perdida.append(perdida_media)

            if verbose and (epoca + 1) % 30 == 0:
                print(f"  │  Época {epoca+1:4d}/{epocas} │ Pérdida: {perdida_media:.4f}")

        if verbose:
            print("  └─ Entrenamiento completado ✓\n")
        self.entrenado = True

    def predecir(self, x):
        """Predicción"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        probs = self.adelante(x)
        return probs[0]

    def generar_datos_sinteticos(self, n_por_clase=80):
        """Genera datos sintéticos de entrenamiento"""
        np.random.seed(123)
        X_lista, y_lista = [], []

        for idx, clase in enumerate(CLASES_TUMOR):
            for _ in range(n_por_clase):
                p = np.array(PATRONES_CLASIFICACION[clase])
                d = np.array(DESVIACIONES_CLASIFICACION[clase])
                muestra = np.clip(np.random.normal(p, d), 0, 1)
                X_lista.append(muestra)
                etiqueta = np.zeros(5)
                etiqueta[idx] = 1
                y_lista.append(etiqueta)

        X = np.array(X_lista)
        y = np.array(y_lista)
        idx_barajado = np.random.permutation(len(X))
        return X[idx_barajado], y[idx_barajado]

    CLASES = CLASES_TUMOR
    GRADOS = GRADOS_WHO


class RedNeuronalRiesgoQuirurgico:
    """
    Red neuronal para estimar score de riesgo quirúrgico (0-100)
    Considera: localización, tamaño, tipo histológico, estado funcional
    Arquitectura: 8 → 24 → 12 → 1
    """

    def __init__(self):
        np.random.seed(RED_RIESGO['seed'])
        self.capas = [
            CapaDensa(8, 24, 'relu'),
            CapaDensa(24, 12, 'relu'),
            CapaDensa(12, 1, 'sigmoide'),
        ]
        self.historial_perdida = []
        self.entrenado = False

    def adelante(self, x):
        """Propagación hacia adelante"""
        for capa in self.capas:
            x = capa.adelante(x)
        return x

    def calcular_perdida_emc(self, y_pred, y_real):
        """Error cuadrático medio"""
        return np.mean((y_pred - y_real)**2)

    def atras(self, y_pred, y_real, lr=0.001):
        """Retropropagación"""
        grad = 2 * (y_pred - y_real) / y_real.shape[0]
        for capa in reversed(self.capas):
            grad = capa.atras(grad, lr)

    def entrenar(self, X, y, epocas=120, lr=0.001, verbose=True):
        """Entrenamiento"""
        if verbose:
            print("  ┌─ Iniciando entrenamiento de Red de Riesgo Quirúrgico...")

        for epoca in range(epocas):
            y_pred = self.adelante(X)
            perdida = self.calcular_perdida_emc(y_pred, y)
            self.historial_perdida.append(perdida)
            self.atras(y_pred, y, lr)

            if verbose and (epoca + 1) % 30 == 0:
                print(f"  │  Época {epoca+1:4d}/{epocas} │ EMC: {perdida:.6f}")

        if verbose:
            print("  └─ Entrenamiento completado ✓\n")
        self.entrenado = True

    def predecir(self, x):
        """Predicción de riesgo (0-100)"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return float(self.adelante(x)[0, 0]) * 100

    def generar_datos_riesgo(self, n=400):
        """
        Genera datos sintéticos de riesgo
        Características: malignidad, tamaño, profundidad, área elocuente, 
                        proximidad vasos, edad, estado funcional, edema
        """
        np.random.seed(77)
        X = np.random.rand(n, 8)

        # Score de riesgo: suma ponderada de factores
        y = (
            0.25 * X[:, 0] +
            0.20 * X[:, 1] +
            0.15 * X[:, 2] +
            0.20 * X[:, 3] +
            0.10 * X[:, 4] +
            0.05 * X[:, 5] +
            -0.10 * X[:, 6] +
            0.05 * X[:, 7]
        )
        y = np.clip(y + np.random.normal(0, 0.03, n), 0, 1)
        return X, y.reshape(-1, 1)
