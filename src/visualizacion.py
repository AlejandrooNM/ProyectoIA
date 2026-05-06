"""
Módulo de visualización de resultados
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from datetime import datetime

from .configuraciones import COLORES, CURVAS_SUPERVIVENCIA


class VisualizadorResultados:
    """Sistema de visualización de resultados del análisis"""

    COLORES = COLORES

    @classmethod
    def generar_reporte_visual(cls, resultado, historial_cls, historial_rsg):
        """Genera el panel de resultados completo"""

        fig = plt.figure(figsize=(20, 14), facecolor=cls.COLORES['fondo'])
        gs = GridSpec(3, 5, figure=fig,
                      hspace=0.45, wspace=0.40,
                      left=0.04, right=0.97,
                      top=0.93, bottom=0.05)

        fig.patch.set_facecolor(cls.COLORES['fondo'])

        # ── Título del sistema ──
        fig.text(
            0.5, 0.965,
            'SISTEMA DE ANÁLISIS DE TUMORES CEREBRALES  ·  RED NEURONAL',
            ha='center', va='top', fontsize=14, fontweight='bold',
            color=cls.COLORES['acento'], fontfamily='monospace',
            path_effects=[pe.withStroke(linewidth=3, foreground='#001122')]
        )
        fig.text(
            0.5, 0.945,
            f'Reporte generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  '
            f'│  Paciente: {resultado["edad_paciente"]} años  '
            f'│  Karnofsky: {resultado["karnofsky"]}',
            ha='center', va='top', fontsize=9,
            color=cls.COLORES['subtexto'], fontfamily='monospace'
        )

        # ─────────────────────────────────────────────
        # FILA 1: Imágenes cerebrales
        # ─────────────────────────────────────────────
        ax_base = fig.add_subplot(gs[0, 0])
        ax_tumor = fig.add_subplot(gs[0, 1])

        cmap_mri = LinearSegmentedColormap.from_list(
            'mri', ['#000000', '#1a1a3a', '#4a3060', '#a06090', '#e0c080', '#ffffff']
        )

        ax_base.imshow(resultado['img_base'], cmap=cmap_mri, vmin=0, vmax=1)
        ax_base.set_title('MRI CEREBRAL BASE', fontsize=8, color=cls.COLORES['acento'],
                           fontfamily='monospace', pad=5)
        ax_base.axis('off')
        cls._estilizar_ax(ax_base)

        ax_tumor.imshow(resultado['img_tumor'], cmap=cmap_mri, vmin=0, vmax=1)
        tx, ty, radio = resultado['posicion_tumor']
        circulo = plt.Circle((ty, tx), radio, color=cls.COLORES['acento2'],
                              fill=False, linewidth=1.5, linestyle='--')
        ax_tumor.add_patch(circulo)
        ax_tumor.plot(ty, tx, '+', color=cls.COLORES['acento'], ms=8, mew=1.5)
        ax_tumor.set_title('REGIÓN TUMORAL DETECTADA', fontsize=8,
                            color=cls.COLORES['acento2'], fontfamily='monospace', pad=5)
        ax_tumor.axis('off')
        cls._estilizar_ax(ax_tumor)

        # ─────────────────────────────────────────────
        # FILA 1: Probabilidades
        # ─────────────────────────────────────────────
        ax_prob = fig.add_subplot(gs[0, 2])
        cls._estilizar_ax(ax_prob)

        clases = list(resultado['probabilidades'].keys())
        probs = list(resultado['probabilidades'].values())
        colores_barras = [
            cls.COLORES['acento'] if c == resultado['tipo_predicho']
            else cls.COLORES['subtexto']
            for c in clases
        ]

        barras = ax_prob.barh(
            range(len(clases)), [p * 100 for p in probs],
            color=colores_barras, height=0.6, alpha=0.85
        )
        for i, (barra, p) in enumerate(zip(barras, probs)):
            ax_prob.text(
                min(p * 100 + 1, 95), i,
                f'{p * 100:.1f}%',
                va='center', fontsize=7, color=cls.COLORES['texto'],
                fontfamily='monospace'
            )

        ax_prob.set_yticks(range(len(clases)))
        ax_prob.set_yticklabels(
            [c[:12] for c in clases],
            fontsize=7.5, color=cls.COLORES['texto'], fontfamily='monospace'
        )
        ax_prob.set_xlabel('Probabilidad (%)', fontsize=7,
                            color=cls.COLORES['subtexto'], fontfamily='monospace')
        ax_prob.set_xlim(0, 108)
        ax_prob.set_title('CLASIFICACIÓN HISTOLÓGICA', fontsize=8,
                           color=cls.COLORES['acento'], fontfamily='monospace', pad=5)
        ax_prob.tick_params(colors=cls.COLORES['subtexto'], labelsize=7)

        # ─────────────────────────────────────────────
        # FILA 1: Gauge de riesgo
        # ─────────────────────────────────────────────
        ax_gauge = fig.add_subplot(gs[0, 3], projection='polar')
        cls._dibujar_gauge(ax_gauge, resultado['score_riesgo'])

        # ─────────────────────────────────────────────
        # FILA 1: Métricas clave
        # ─────────────────────────────────────────────
        ax_metricas = fig.add_subplot(gs[0, 4])
        cls._estilizar_ax(ax_metricas)
        ax_metricas.axis('off')

        metricas = [
            ('TIPO TUMORAL', resultado['tipo_predicho'], cls.COLORES['acento']),
            ('GRADO WHO', resultado['grado_who'], cls.COLORES['amarillo']),
            ('DIÁMETRO', f"{resultado['diametro_mm']:.1f} mm", cls.COLORES['texto']),
            ('VOLUMEN', f"{resultado['volumen_mm3']:.0f} mm³", cls.COLORES['texto']),
            ('RIESGO QX', f"{resultado['score_riesgo']:.1f}/100",
             cls._color_riesgo(resultado['score_riesgo'])),
            ('CONFIANZA IA', f"{resultado['confianza'] * 100:.1f}%", cls.COLORES['verde']),
            ('PRONÓSTICO', resultado['pronostico'][:18], cls.COLORES['amarillo']),
        ]

        for i, (etiqueta, valor, color) in enumerate(metricas):
            y = 0.95 - i * 0.135
            ax_metricas.text(0.02, y, etiqueta, transform=ax_metricas.transAxes,
                              fontsize=6.5, color=cls.COLORES['subtexto'],
                              fontfamily='monospace')
            ax_metricas.text(0.02, y - 0.055, valor, transform=ax_metricas.transAxes,
                              fontsize=8.5, color=color, fontweight='bold',
                              fontfamily='monospace')

        ax_metricas.set_title('MÉTRICAS CLÍNICAS', fontsize=8,
                               color=cls.COLORES['acento'], fontfamily='monospace', pad=5)

        # ─────────────────────────────────────────────
        # FILA 2: Curvas de entrenamiento
        # ─────────────────────────────────────────────
        ax_train1 = fig.add_subplot(gs[1, 0:2])
        cls._estilizar_ax(ax_train1)

        suavizado_cls = gaussian_filter(historial_cls, sigma=3)
        suavizado_rsg = gaussian_filter(
            [h * 100 for h in historial_rsg], sigma=3
        )

        ax_train1.plot(historial_cls, color=cls.COLORES['subtexto'],
                       alpha=0.3, linewidth=0.8)
        ax_train1.plot(suavizado_cls, color=cls.COLORES['acento'],
                       linewidth=2.0, label='Red Clasificación')

        ax_twin = ax_train1.twinx()
        ax_twin.plot(suavizado_rsg, color=cls.COLORES['verde'],
                     linewidth=1.5, linestyle='--', label='Red Riesgo (×100)', alpha=0.8)
        ax_twin.tick_params(colors=cls.COLORES['subtexto'], labelsize=7)

        ax_train1.set_title('CONVERGENCIA DEL ENTRENAMIENTO', fontsize=8,
                             color=cls.COLORES['acento'], fontfamily='monospace', pad=5)
        ax_train1.set_xlabel('Época', fontsize=7, color=cls.COLORES['subtexto'],
                              fontfamily='monospace')
        ax_train1.set_ylabel('Pérdida', fontsize=7,
                              color=cls.COLORES['subtexto'], fontfamily='monospace')
        ax_train1.tick_params(colors=cls.COLORES['subtexto'], labelsize=7)

        lines1, labels1 = ax_train1.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_train1.legend(
            lines1 + lines2, labels1 + labels2,
            fontsize=7, facecolor=cls.COLORES['panel'],
            labelcolor=cls.COLORES['texto'], loc='upper right',
            framealpha=0.7
        )

        # ─────────────────────────────────────────────
        # FILA 2: Mapa de calor
        # ─────────────────────────────────────────────
        ax_heat = fig.add_subplot(gs[1, 2])
        cls._estilizar_ax(ax_heat)

        mask_region = np.zeros_like(resultado['img_tumor'])
        tam = 64
        for i in range(tam):
            for j in range(tam):
                d = np.sqrt((i - tx)**2 + (j - ty)**2)
                if d < radio * 1.8:
                    mask_region[i, j] = np.exp(-0.5 * (d / (radio * 0.6))**2)

        ax_heat.imshow(resultado['img_tumor'], cmap='gray', alpha=0.5)
        cmap_heat = LinearSegmentedColormap.from_list(
            'heat', ['#000000', '#ff000000', '#ff4400', '#ffaa00', '#ffff00']
        )
        ax_heat.imshow(mask_region, cmap=cmap_heat, alpha=0.7)
        ax_heat.set_title('MAPA DENSIDAD TUMORAL', fontsize=8,
                           color=cls.COLORES['amarillo'], fontfamily='monospace', pad=5)
        ax_heat.axis('off')

        # ─────────────────────────────────────────────
        # FILA 2: Radar
        # ─────────────────────────────────────────────
        ax_radar = fig.add_subplot(gs[1, 3], projection='polar')
        cls._dibujar_radar(ax_radar, resultado['features'])

        # ─────────────────────────────────────────────
        # FILA 2: Supervivencia
        # ─────────────────────────────────────────────
        ax_sv = fig.add_subplot(gs[1, 4])
        cls._estilizar_ax(ax_sv)
        cls._dibujar_curva_supervivencia(ax_sv, resultado['tipo_predicho'])

        # ─────────────────────────────────────────────
        # FILA 3: Recomendaciones
        # ─────────────────────────────────────────────
        ax_recs = fig.add_subplot(gs[2, :])
        cls._estilizar_ax(ax_recs)
        ax_recs.axis('off')

        ax_recs.set_title('RECOMENDACIONES CLÍNICAS  ·  PLAN DE MANEJO',
                           fontsize=9, color=cls.COLORES['acento'],
                           fontfamily='monospace', pad=6)

        recs = resultado['recomendaciones']
        n_cols = 3
        items_por_col = (len(recs) + n_cols - 1) // n_cols

        for idx, rec in enumerate(recs):
            col = idx // items_por_col
            fila = idx % items_por_col
            x = 0.02 + col * 0.34
            y = 0.80 - fila * 0.22

            color_punto = cls._color_riesgo(resultado['score_riesgo']) if idx == 0 else cls.COLORES['acento']
            ax_recs.text(x, y, '▸', transform=ax_recs.transAxes,
                         fontsize=9, color=color_punto, va='top')
            ax_recs.text(x + 0.018, y, rec, transform=ax_recs.transAxes,
                         fontsize=8, color=cls.COLORES['texto'],
                         fontfamily='monospace', va='top')

        ax_recs.text(
            0.5, 0.10,
            f'Supervivencia estimada: {resultado["supervivencia_estimada"]}',
            transform=ax_recs.transAxes, ha='center', fontsize=9,
            color=cls.COLORES['amarillo'], fontfamily='monospace',
            fontweight='bold'
        )

        fig.text(
            0.5, 0.01,
            '⚠  SISTEMA DE APOYO A LA DECISIÓN — No reemplaza el juicio clínico del especialista  ⚠',
            ha='center', fontsize=8, color='#ff6b6b',
            fontfamily='monospace', alpha=0.7
        )

        return fig

    @classmethod
    def _estilizar_ax(cls, ax):
        """Estiliza un axis"""
        ax.set_facecolor(cls.COLORES['panel'])
        for spine in ax.spines.values():
            spine.set_edgecolor(cls.COLORES['borde'])
            spine.set_linewidth(0.8)
        ax.tick_params(colors=cls.COLORES['subtexto'])

    @classmethod
    def _color_riesgo(cls, score):
        """Retorna color según nivel de riesgo"""
        if score >= 70:
            return '#ff4444'
        elif score >= 40:
            return '#ffd166'
        else:
            return '#00ff88'

    @classmethod
    def _dibujar_gauge(cls, ax, score):
        """Medidor de riesgo semicircular"""
        ax.set_facecolor(cls.COLORES['panel'])
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_ylim(0, 1)

        theta_verde = np.linspace(0, np.pi * 0.4, 100)
        theta_amarillo = np.linspace(np.pi * 0.4, np.pi * 0.7, 100)
        theta_rojo = np.linspace(np.pi * 0.7, np.pi, 100)

        for theta, color in [(theta_verde, '#00ff88'),
                              (theta_amarillo, '#ffd166'),
                              (theta_rojo, '#ff4444')]:
            ax.fill_between(theta, 0.65, 0.85, color=color, alpha=0.7)

        angulo_aguja = np.pi * (1 - score / 100)
        ax.annotate('', xy=(angulo_aguja, 0.7), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='white',
                                     lw=2, mutation_scale=12))
        ax.set_rticks([])
        ax.set_xticks([])
        ax.set_title(
            f'RIESGO QX\n{score:.0f}/100',
            fontsize=8, color=cls._color_riesgo(score),
            fontfamily='monospace', pad=10
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    @classmethod
    def _dibujar_radar(cls, ax, features):
        """Gráfico de radar"""
        labels = ['Intensidad', 'Heterog.', 'Simetría',
                  'Gradiente', 'Localiz.', 'Volumen',
                  'Q25', 'Q75', 'IQR', 'Entropía']
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        vals = features.tolist()
        vals += vals[:1]

        ax.set_facecolor(cls.COLORES['panel'])
        ax.plot(angles, vals, color=cls.COLORES['acento'], linewidth=1.5, alpha=0.9)
        ax.fill(angles, vals, color=cls.COLORES['acento'], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=6, color=cls.COLORES['texto'],
                            fontfamily='monospace')
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=5,
                            color=cls.COLORES['subtexto'])
        ax.set_ylim(0, 1)
        ax.set_title('PERFIL MORFOLÓGICO', fontsize=8,
                     color=cls.COLORES['acento'], fontfamily='monospace', pad=10)
        ax.grid(color=cls.COLORES['borde'], alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(cls.COLORES['borde'])

    @classmethod
    def _dibujar_curva_supervivencia(cls, ax, tipo_tumor):
        """Curva de supervivencia Kaplan-Meier"""
        curvas = CURVAS_SUPERVIVENCIA
        datos = curvas.get(tipo_tumor, curvas['Astrocitoma'])
        tiempo = np.array(datos['tiempo'])
        sv = np.array(datos['supervivencia'])

        ax.step(tiempo, sv, where='post', color=cls.COLORES['acento'],
                linewidth=2, label=tipo_tumor)
        ax.fill_between(tiempo, sv * 0.85, sv * 1.05, alpha=0.15,
                        color=cls.COLORES['acento'], step='post')
        ax.axhline(0.5, color=cls.COLORES['acento2'], linestyle=':', linewidth=1,
                   alpha=0.7, label='SV 50%')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Meses', fontsize=7, color=cls.COLORES['subtexto'],
                      fontfamily='monospace')
        ax.set_ylabel('Supervivencia', fontsize=7, color=cls.COLORES['subtexto'],
                      fontfamily='monospace')
        ax.set_title('CURVA DE SUPERVIVENCIA', fontsize=8,
                     color=cls.COLORES['acento'], fontfamily='monospace', pad=5)
        ax.legend(fontsize=6.5, facecolor=cls.COLORES['panel'],
                  labelcolor=cls.COLORES['texto'], framealpha=0.6)
        ax.tick_params(colors=cls.COLORES['subtexto'], labelsize=7)
