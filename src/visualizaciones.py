# -------------------------
# VISUALIZACIONES: Funciones de visualización de datos y resultados
# -------------------------
import plotly.graph_objects as go

def visualizar_historico_predicciones(df_historico, df_prediccion, columna_target='base_imponible', columna_fecha='week_start', ax=None):
    """
    Grafica los valores históricos y todas las predicciones en un gráfico interactivo Plotly.
    Args:
        df_historico: DataFrame con los datos históricos (debe incluir columna de fecha y target)
        df_prediccion: DataFrame con las predicciones (puede incluir varias fechas y valores)
        columna_target: Nombre de la columna objetivo en el histórico
        columna_fecha: Nombre de la columna de fechas
    Returns:
        fig: objeto Plotly Figure
    """
    fig = go.Figure()
    # Histórico
    fig.add_trace(go.Scatter(
        x=df_historico[columna_fecha],
        y=df_historico[columna_target],
        mode='lines+markers',
        name='Histórico',
        hovertemplate='Fecha: %{x}<br>Histórico: %{y}<extra></extra>'
    ))
    # Predicciones (pueden ser varias)
    fig.add_trace(go.Scatter(
        x=df_prediccion[columna_fecha],
        y=df_prediccion['predicted_base_imponible'],
        mode='lines+markers',
        name='Predicción',
        line=dict(color='red', dash='dash'),
        marker=dict(symbol='x', size=10),
        hovertemplate='Fecha: %{x}<br>Predicción: %{y}<extra></extra>'
    ))
    fig.update_layout(
        title='Histórico y Predicciones de Ventas de Bollería',
        xaxis_title='Fecha',
        yaxis_title='Base Imponible',
        legend=dict(x=0, y=1),
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
