"""
Módulo de generación de reportes
"""

from datetime import datetime


def imprimir_reporte_clinico(resultado):
    """Imprime el reporte estructurado en consola"""

    sep  = "═" * 65
    sep2 = "─" * 65

    print()
    print(sep)
    print("  REPORTE CLÍNICO DE ANÁLISIS NEURAL")
    print(f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    print(f"\n  DATOS DEL PACIENTE")
    print(sep2)
    print(f"  Edad              : {resultado['edad_paciente']} años")
    print(f"  Escala Karnofsky  : {resultado['karnofsky']}/100")

    print(f"\n  HALLAZGOS DE IMAGEN")
    print(sep2)
    print(f"  Tipo tumoral predicho : {resultado['tipo_predicho']}")
    print(f"  Tipo real (referencia): {resultado['tipo_real']}")
    print(f"  Clasificación WHO     : {resultado['grado_who']}")
    print(f"  Confianza del modelo  : {resultado['confianza'] * 100:.1f}%")

    print(f"\n  MORFOMETRÍA")
    print(sep2)
    print(f"  Diámetro máximo  : {resultado['diametro_mm']:.1f} mm")
    print(f"  Volumen estimado : {resultado['volumen_mm3']:.0f} mm³")
    print(f"  Posición (voxel) : ({resultado['posicion_tumor'][0]}, {resultado['posicion_tumor'][1]})")

    print(f"\n  PROBABILIDADES POR TIPO")
    print(sep2)
    for tipo, prob in sorted(resultado['probabilidades'].items(),
                              key=lambda x: -x[1]):
        barra = '█' * int(prob * 30) + '░' * (30 - int(prob * 30))
        marca = ' ◄' if tipo == resultado['tipo_predicho'] else ''
        print(f"  {tipo:20s} {barra} {prob * 100:5.1f}%{marca}")

    score = resultado['score_riesgo']
    nivel = "ALTO 🔴" if score >= 70 else ("MODERADO 🟡" if score >= 40 else "BAJO 🟢")
    print(f"\n  RIESGO QUIRÚRGICO")
    print(sep2)
    print(f"  Score : {score:.1f}/100  →  {nivel}")

    print(f"\n  PRONÓSTICO")
    print(sep2)
    print(f"  Evaluación          : {resultado['pronostico']}")
    print(f"  Supervivencia estim.: {resultado['supervivencia_estimada']}")

    print(f"\n  RECOMENDACIONES CLÍNICAS")
    print(sep2)
    for i, rec in enumerate(resultado['recomendaciones'], 1):
        print(f"  {i:2d}. {rec}")

    print()
    print(sep)
    print("  ⚠  AVISO LEGAL: Sistema de apoyo diagnóstico. No reemplaza")
    print("     la evaluación por neurocirujano y neurooncólogo certificados.")
    print(sep)
    print()
