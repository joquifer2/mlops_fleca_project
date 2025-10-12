# Propuesta EDA para Forecasting de Ventas en Hostelería Turística (Cafetería + Panadería)

## 🎯 Objetivo del EDA

- Comprender la dinámica temporal de las ventas: tendencias, estacionalidades y patrones recurrentes.
- Identificar variables exógenas o factores externos que influyen en la demanda.
- Validar la estabilidad y calidad de la serie temporal para garantizar la fiabilidad de un modelo de predicción.
- Detectar oportunidades de segmentación que mejoren la precisión del forecasting.

## ✅ 1. Análisis Exploratorio Temporal

### a) Evolución global de ventas

- Ventas agregadas por:
  - Día
  - Semana
  - Mes
- Comparativa interanual: 2023 vs 2024 vs 2025 (hasta junio).
- Descomposición en:
  - Tendencia
  - Estacionalidad
  - Residuo o irregularidad

### b) Granularidad óptima para forecasting

- Determinar si es mejor predecir:
  - Por día
  - Por semana
  - Por franja horaria (si la variabilidad intra-día es relevante).

## ✅ 2. Análisis de Estacionalidad y Ciclos

### a) Estacionalidad anual

- Incrementos en:
  - Verano
  - Puentes y festivos
  - Semana Santa
  - Navidad
  - Diferencias entre meses de temporada alta turística vs baja.

### b) Estacionalidad semanal

- Comportamiento en fines de semana vs días laborales.

### c) Estacionalidad diaria / horaria

- Análisis por franja horaria para detectar picos y valles.

## ✅ 3. Análisis por Segmentos Estratégicos

### a) Por categoría de producto

- Panadería vs Cafetería vs Bollería.
- Comportamientos estacionales diferenciados.

### b) Por tipo de cliente

- Si es posible identificar turistas vs locales (de forma directa o por proxies: forma de pago, tickets más elevados, etc.).

### c) Ticket medio vs número de tickets

- Separar la evolución:
  - Número de ventas
  - Importe promedio por ticket
- Esto permite modelar unidades y facturación por separado.

## ✅ 4. Estabilidad y Calidad de la Serie

- Datos faltantes o inconsistentes.
- Cambios estructurales:
  - Horarios de apertura
  - Cambios de precios
  - Reformas, cierres temporales.
- Transformaciones necesarias:
  - Estabilizar la varianza (logaritmo, Box-Cox).
  - Estacionarizar si hay tendencia o estacionalidad.

## ✅ 5. Análisis Estadístico Temporal

- **ACF/PACF**: determinar autocorrelaciones y lags relevantes.
- **Prueba de Dickey-Fuller**: test de estacionariedad.
- **Rolling means y rolling std**: evaluar estabilidad en ventanas móviles.
- **Descomposición STL**: desglosar la serie en tendencia, estacionalidad y residuo.

## ✅ 6. Variables Exógenas (Features externas)

- Festivos locales y nacionales.
- Vacaciones escolares.
- Eventos locales y festivales.
- Clima y temperatura media diaria (relevante para productos fríos o calientes).
- Calendario escolar (puede alterar los flujos de turistas).
- Variables dummy para:
  - Temporada alta/baja.
  - Fines de semana.

## ✅ 7. Identificación de Outliers y Eventos Especiales

- Días con ventas atípicas.
- Eventos como ferias, festivales, o deportivos.
- Identificar si incluirlos o modelarlos por separado.

## ✅ 8. Output del EDA para Forecasting

- Diagnóstico de la serie temporal:
  - Tendencia
  - Estacionalidades (anual, semanal, diaria)
  - Estabilidad
- Definición de:
  - Granularidad del forecasting
  - Variables exógenas relevantes
  - Necesidad de segmentación por categoría o tipo de cliente
- Documentación de outliers relevantes para el negocio.

## ✅ 9. Opcional: Clusterización de días o semanas

- Para detectar patrones similares en:
  - Semanas de temporada alta
  - Semanas bajas o intermedias
- Esto permite hacer forecasting diferenciado o condicional.

## 🚀 Siguientes pasos

- Puedo generarte directamente:
  - Un template de código EDA en Python adaptado a series temporales.
  - Un informe estructurado en formato presentación o markdown para que documentes el análisis.
  - Una checklist de features externas recomendadas para el modelo.