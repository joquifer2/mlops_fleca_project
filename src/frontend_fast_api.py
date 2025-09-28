# -------------------------
# CONFIGURACIÓN INICIAL Y LIBRERÍAS
# -------------------------

import os
import streamlit as st
from datetime import datetime
import pandas as pd
from api_client import llamar_api_prediccion, obtener_datos_grafico
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# CONFIGURACIÓN DE LA INTERFAZ STREAMLIT
# -------------------------
st.set_page_config(layout="wide")

st.title('🥐 Predicción semanal de ventas de bollería')
st.header('by Jordi Quiroga')
st.markdown("---")

# Debug opcional: ver qué API_URL está leyendo la UI
with st.expander("🔧 Información de configuración"):
    st.caption(f"API_URL = {os.getenv('API_URL', '(sin definir; usando fallback)')}")
    st.caption(f"HTTP_PROXY={os.getenv('HTTP_PROXY')}  HTTPS_PROXY={os.getenv('HTTPS_PROXY')}  ALL_PROXY={os.getenv('ALL_PROXY')}")
    st.caption(f"http_proxy={os.getenv('http_proxy')}  https_proxy={os.getenv('https_proxy')}  all_proxy={os.getenv('all_proxy')}")

# -------------------------
# PREDICCIÓN DESDE LA API
# -------------------------
st.subheader('📊 Obtener predicción')

if st.button('🚀 Generar predicción', type='primary'):
    with st.spinner('Obteniendo predicción desde la API...'):
        try:
            timestamp = datetime.now()
            resultado_api = llamar_api_prediccion(timestamp)
            
            # Mostrar resultado principal
            st.success('✅ Predicción obtenida exitosamente!')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="📅 Semana de predicción", 
                    value=resultado_api['week_start'][0]
                )
            
            with col2:
                st.metric(
                    label="💰 Valor predicho", 
                    value=f"{resultado_api['prediction'][0]:.2f} €"
                )
            
            # Mostrar DataFrame de resultado
            df_resultado = pd.DataFrame({
                'Semana': [resultado_api['week_start'][0]],
                'Predicción (€)': [f"{resultado_api['prediction'][0]:.2f}"]
            })
            
            st.subheader('📋 Detalle de la predicción')
            st.dataframe(df_resultado, use_container_width=True)
            
            # Debug opcional
            with st.expander("🔍 Respuesta completa de la API (debug)"):
                st.json(resultado_api)
                
        except Exception as e:
            st.error(f"❌ Error al obtener la predicción: {str(e)}")
            st.info("💡 Verifica que la API esté funcionando correctamente.")

# -------------------------
# GRÁFICO HISTÓRICO Y PREDICCIONES
# -------------------------
st.markdown("---")
st.subheader('📈 Gráfico comparativo')

if st.button('📊 Mostrar gráfico histórico vs predicciones'):
    with st.spinner('Obteniendo datos históricos desde la API...'):
        try:
            datos_grafico = obtener_datos_grafico()
            
            # Crear el gráfico con Plotly
            fig = go.Figure()
            
            # Datos históricos (reales)
            if datos_grafico['historical']:
                df_hist = pd.DataFrame(datos_grafico['historical'])
                df_hist['week_start'] = pd.to_datetime(df_hist['week_start'])
                
                fig.add_trace(go.Scatter(
                    x=df_hist['week_start'],
                    y=df_hist['base_imponible'],
                    mode='lines+markers',
                    name='Valores Reales',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
            
            # Predicciones históricas
            if datos_grafico['predictions']:
                df_pred = pd.DataFrame(datos_grafico['predictions'])
                df_pred['week_start'] = pd.to_datetime(df_pred['week_start'])
                
                fig.add_trace(go.Scatter(
                    x=df_pred['week_start'],
                    y=df_pred['predicted_base_imponible'],
                    mode='lines+markers',
                    name='Predicciones',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=4, symbol='x')
                ))
            
            # Configurar el gráfico
            fig.update_layout(
                title='📈 Histórico de Ventas vs Predicciones',
                xaxis_title='Fecha (Semanas)',
                yaxis_title='Base Imponible (€)',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template='plotly_white'
            )
            
            # Mostrar el gráfico
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar estadísticas básicas
            col1, col2 = st.columns(2)
            
            if datos_grafico['historical']:
                with col1:
                    st.metric(
                        "📊 Total semanas históricas",
                        len(datos_grafico['historical'])
                    )
            
            if datos_grafico['predictions']:
                with col2:
                    st.metric(
                        "🔮 Total predicciones",
                        len(datos_grafico['predictions'])
                    )
                    
        except Exception as e:
            st.error(f"❌ Error al obtener los datos del gráfico: {str(e)}")
            st.info("💡 Verifica que la API y Hopsworks estén funcionando correctamente.")

# -------------------------
# INFORMACIÓN ADICIONAL
# -------------------------
st.markdown("---")
st.subheader('ℹ️ Información del modelo')

st.info("""
**📈 Modelo de predicción de ventas semanales**

Este sistema predice las ventas de productos de bollería para la próxima semana utilizando:
- 🤖 Algoritmos de Machine Learning (XGBoost)
- 📊 Datos históricos de ventas
- 🔄 Pipeline MLOps completo con Hopsworks
- ⚡ API REST para predicciones en tiempo real

**🏗️ Arquitectura:**
- **Frontend (Streamlit):** Interfaz de usuario para visualizar predicciones
- **API (FastAPI):** Lógica de negocio y conexión con el modelo
- **Hopsworks:** Feature Store y Model Registry para MLOps

**🌐 Despliegue:**
- API desplegada en Render
- Frontend disponible en Streamlit Cloud y Render
""")

st.markdown("---")
st.markdown("*Proyecto MLOps - Máster ML Engineer Nodd3r*")

