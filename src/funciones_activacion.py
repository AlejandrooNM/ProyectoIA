"""
Funciones de activación y utilidades de redes neuronales
"""

import numpy as np


def relu(x):
    """Función de activación ReLU"""
    return np.maximum(0, x)


def derivada_relu(x):
    """Derivada de ReLU"""
    return (x > 0).astype(float)


def sigmoide(x):
    """Función sigmoide con clipping para estabilidad numérica"""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def derivada_sigmoide(x):
    """Derivada de sigmoide"""
    s = sigmoide(x)
    return s * (1 - s)


def softmax(x):
    """Softmax estable numéricamente"""
    x_desplazado = x - np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x_desplazado)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def relu_con_fugas(x, alpha=0.01):
    """Leaky ReLU para evitar dead neurons"""
    return np.where(x > 0, x, alpha * x)


def tanh_activacion(x):
    """Función de activación tanh"""
    return np.tanh(x)


def obtener_funcion_activacion(nombre):
    """Retorna la función de activación por nombre"""
    activaciones = {
        'relu': relu,
        'sigmoide': sigmoide,
        'softmax': softmax,
        'relu_con_fugas': relu_con_fugas,
        'tanh': tanh_activacion,
    }
    return activaciones.get(nombre, lambda x: x)


def obtener_derivada_activacion(nombre):
    """Retorna la derivada de la función de activación"""
    derivadas = {
        'relu': derivada_relu,
        'sigmoide': derivada_sigmoide,
        'tanh': lambda x: 1 - tanh_activacion(x)**2,
        'relu_con_fugas': lambda x: np.where(x > 0, 1.0, 0.01),
        'softmax': lambda x: x,
        'lineal': lambda x: np.ones_like(x),
    }
    return derivadas.get(nombre, lambda x: np.ones_like(x))
