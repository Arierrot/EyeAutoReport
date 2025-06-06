# EyeAutoReport

# CorneaSegmentation

Desarrollé un método automatizado para segmentar capas corneales en imágenes AS-OCT (Tomografía de Coherencia Óptica del Segmento Anterior), con el objetivo de analizar cuantitativamente la relación lente-córnea en pacientes con córnea irregular.

La herramienta permite la obtención rápida de informes gráficos con superposición de capas, gradientes de distancia, histogramas y tablas.

<p align="center">
  <img src="reports/im3_informe.pdf">
</p>

## Descripción

Este repositorio contiene un pipeline completo para:

1. **Preprocesamiento**: filtrado del ruido y realce de bordes horizontales.
2. **Segmentación**: detección automática de las tres capas más relevantes en la imagen.
3. **Medición**: cálculo columna a columna de las distancias verticales entre capas.
4. **Estadística**: obtención de métricas (media, std, simetría lateral, normalidad...).
5. **Visualización**: creación de informes gráficos con superposición de capas, gradientes de distancia, histogramas y tablas.

## Requisitos

El entorno fue probado con las siguientes versiones:

```txt
matplotlib==3.10.3
numpy==2.2.6
opencv_contrib_python==4.11.0.86
pandas==2.3.0
scipy==1.15.3
seaborn==0.13.2
skimage==0.0
