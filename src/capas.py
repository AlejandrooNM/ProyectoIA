"""
Capas de redes neuronales (Densa, Convolucional, Pooling)
"""

import numpy as np
from .funciones_activacion import (
    relu, sigmoide, tanh_activacion, relu_con_fugas, softmax,
    derivada_relu, derivada_sigmoide, obtener_derivada_activacion
)


class CapaDensa:
    """Capa completamente conectada con retropropagación"""

    def __init__(self, n_entrada, n_salida, activacion='relu'):
        # Inicialización He para ReLU, Xavier para otros
        if activacion == 'relu':
            escala = np.sqrt(2.0 / n_entrada)
        else:
            escala = np.sqrt(1.0 / n_entrada)

        self.W = np.random.randn(n_entrada, n_salida) * escala
        self.b = np.zeros((1, n_salida))
        self.activacion = activacion

        # Para retropropagación
        self.ultimo_x = None
        self.ultimo_z = None

        # Momentos Adam
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.t = 0

    def adelante(self, x):
        """Propagación hacia adelante"""
        self.ultimo_x = x
        self.ultimo_z = x @ self.W + self.b
        
        if self.activacion == 'relu':
            return relu(self.ultimo_z)
        elif self.activacion == 'sigmoide':
            return sigmoide(self.ultimo_z)
        elif self.activacion == 'tanh':
            return tanh_activacion(self.ultimo_z)
        elif self.activacion == 'softmax':
            return softmax(self.ultimo_z)
        elif self.activacion == 'relu_con_fugas':
            return relu_con_fugas(self.ultimo_z)
        else:
            return self.ultimo_z

    def atras(self, grad_salida, lr=0.001):
        """Retropropagación con optimizador Adam"""
        m = self.ultimo_x.shape[0]

        # Calcular gradiente según función de activación
        if self.activacion == 'relu':
            grad_z = grad_salida * derivada_relu(self.ultimo_z)
        elif self.activacion == 'sigmoide':
            grad_z = grad_salida * derivada_sigmoide(self.ultimo_z)
        elif self.activacion == 'tanh':
            t = tanh_activacion(self.ultimo_z)
            grad_z = grad_salida * (1 - t**2)
        elif self.activacion in ['softmax', 'lineal']:
            grad_z = grad_salida
        elif self.activacion == 'relu_con_fugas':
            grad_z = grad_salida * np.where(self.ultimo_z > 0, 1.0, 0.01)
        else:
            grad_z = grad_salida

        grad_W = self.ultimo_x.T @ grad_z / m
        grad_b = np.mean(grad_z, axis=0, keepdims=True)
        grad_x = grad_z @ self.W.T

        # Actualización Adam
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.t += 1
        self.m_W = beta1 * self.m_W + (1 - beta1) * grad_W
        self.v_W = beta2 * self.v_W + (1 - beta2) * grad_W**2
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * grad_b**2

        m_W_hat = self.m_W / (1 - beta1**self.t)
        v_W_hat = self.v_W / (1 - beta2**self.t)
        m_b_hat = self.m_b / (1 - beta1**self.t)
        v_b_hat = self.v_b / (1 - beta2**self.t)

        self.W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
        self.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

        return grad_x


class CapaConvolucional2D:
    """Capa convolucional 2D para extracción de características espaciales"""

    def __init__(self, n_filtros, tam_kernel=3, activacion='relu'):
        self.n_filtros = n_filtros
        self.k = tam_kernel
        self.activacion = activacion
        self.filtros = None
        self.bias = None
        self.ultimo_x = None

    def _inicializar(self, canales_entrada):
        """Inicialización He"""
        escala = np.sqrt(2.0 / (self.k * self.k * canales_entrada))
        self.filtros = np.random.randn(
            self.n_filtros, canales_entrada, self.k, self.k
        ) * escala
        self.bias = np.zeros(self.n_filtros)

    def adelante(self, x):
        """
        Propagación convolucional
        x: (lote, canales, alto, ancho)
        Retorna: (lote, n_filtros, alto', ancho')
        """
        if self.filtros is None:
            self._inicializar(x.shape[1])

        self.ultimo_x = x
        lote, C, H, W = x.shape
        pad = self.k // 2
        H_sal = H - self.k + 1 + 2 * pad
        W_sal = W - self.k + 1 + 2 * pad

        x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
        salida = np.zeros((lote, self.n_filtros, H_sal, W_sal))

        for f in range(self.n_filtros):
            for i in range(H_sal):
                for j in range(W_sal):
                    region = x_pad[:, :, i:i+self.k, j:j+self.k]
                    salida[:, f, i, j] = (
                        np.sum(region * self.filtros[f], axis=(1,2,3)) + self.bias[f]
                    )

        if self.activacion == 'relu':
            return relu(salida)
        elif self.activacion == 'relu_con_fugas':
            return relu_con_fugas(salida)
        return salida

    def aplanar(self, x):
        """Aplana el tensor para la capa densa"""
        return x.reshape(x.shape[0], -1)


class CapaPooling:
    """Max Pooling 2D para reducir dimensiones"""

    def __init__(self, tam=2):
        self.tam = tam
        self.ultimo_x = None
        self.mascara = None

    def adelante(self, x):
        """Propagación pooling"""
        self.ultimo_x = x
        lote, C, H, W = x.shape
        H_sal = H // self.tam
        W_sal = W // self.tam

        salida = np.zeros((lote, C, H_sal, W_sal))
        for i in range(H_sal):
            for j in range(W_sal):
                region = x[:, :,
                           i*self.tam:(i+1)*self.tam,
                           j*self.tam:(j+1)*self.tam]
                salida[:, :, i, j] = np.max(region, axis=(2,3))

        return salida
