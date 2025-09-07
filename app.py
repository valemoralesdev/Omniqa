import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from pymongo import MongoClient
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

client = MongoClient("mongodb://localhost:27017")
db = client["quality_control"]
collection = db["results"]

def get_data():
    docs = list(collection.find())
    data = []
    for doc in docs:
        metrics = doc.get("metricas", {})
        data.append({
            "UID": doc.get("sop_instance_uid"),
            "Fecha": pd.to_datetime(doc.get("fecha")),
            "Usuario": doc.get("usuario", "Desconocido"),
            "SNR": float(metrics.get("snr_signal", 0)) if metrics.get("snr_signal") is not None else 0,
            "SDNR": float(metrics.get("sdnr", 0)) if metrics.get("sdnr") is not None else 0,
            "Pixel Spacing": metrics.get("pixel_spacing_mm"),
            "MTF_H": metrics.get("mtf_horizontal", {}),
            "MTF_V": metrics.get("mtf_vertical", {}),
            "Equipo": doc.get("origen_equipo", "Desconocido"),
            "Archivo": doc.get("archivo_dicom", "No disponible")
        })

    df = pd.DataFrame(data)
    if df.empty or "Fecha" not in df.columns:
        return pd.DataFrame()  # devuelve vacío sin columnas erróneas

    return df.sort_values("Fecha")


app = dash.Dash(__name__)
app.title = "Control de Calidad"

app.layout = html.Div([
    html.H1("Resultados", style={"textAlign": "center"}),

    dcc.RadioItems(
        id="view_mode",
        options=[
            {"label": "Individual (por UID)", "value": "individual"},
            {"label": "Global (todos los UIDs)", "value": "global"}
        ],
        value="individual",
        labelStyle={"display": "inline-block", "marginRight": "20px"},
        style={"textAlign": "center", "marginBottom": "20px"}
    ),

    dcc.Dropdown(
        id="uid_selector",
        placeholder="Selecciona un UID",
        style={"width": "50%", "margin": "auto"}
    ),

    html.Div(id="info_panel", style={"textAlign": "center", "marginBottom": "20px"}),
    html.Div(id="snr_sdnr_panel"),
    dcc.Graph(id="mtf_chart"),

    dcc.Interval(
        id="interval-refresh",
        interval=10 * 1000,
        n_intervals=0
    )
])

@app.callback(
    Output("uid_selector", "options"),
    Input("interval-refresh", "n_intervals")
)
def update_uid_options(n):
    df = get_data()
    return [{"label": f"{row.UID} - {row.Fecha}", "value": row.UID} for _, row in df.iterrows()]

@app.callback(
    [Output("info_panel", "children"),
     Output("snr_sdnr_panel", "children"),
     Output("mtf_chart", "figure")],
    [Input("uid_selector", "value"),
     Input("view_mode", "value"),
     Input("interval-refresh", "n_intervals")]
)
def update_charts(selected_uid, view_mode, n_intervals):
    df = get_data()

    if df.empty:
        return (
            html.Div("🔴 No hay datos disponibles en la base de datos."),
            html.Div("Esperando que se procese al menos un estudio..."),
            go.Figure().update_layout(title="Sin datos disponibles")
        )

    if view_mode == "individual":
        if not selected_uid or selected_uid not in df["UID"].values:
            return (
                html.Div("⚠️ UID no encontrado o datos incompletos."),
                html.Div("Selecciona un UID válido en el menú desplegable."),
                go.Figure().update_layout(title="Sin datos")
            )

        row = df[df["UID"] == selected_uid].iloc[0]
        snr_val = float(row["SNR"]) if pd.notnull(row["SNR"]) else 0
        sdnr_val = float(row["SDNR"]) if pd.notnull(row["SDNR"]) else 0

        fecha_dt = pd.to_datetime(row["Fecha"])
        info_text = html.Div([
            html.H4("Información del Estudio"),
            html.P(f"UID: {selected_uid}"),
            html.P(f"Fecha: {fecha_dt.date()}"),
            html.P(f"Hora: {fecha_dt.strftime('%H:%M')}"),
            html.P(f"Equipo: {row.get('Equipo', 'Desconocido')}"),
            html.P(f"Archivo: {row.get('Archivo', 'No disponible')}")
        ])

        indicador_estilo = {
            "display": "inline-block",
            "margin": "20px",
            "textAlign": "center"
        }

        snr_box = html.Div([
            html.Div(f"{snr_val:.2f}", style={
                "border": "3px solid #007BFF",
                "backgroundColor": "#E6F0FF",
                "borderRadius": "15px",
                "padding": "30px",
                "fontSize": "50px",
                "color": "black",
                "fontFamily": "Arial Black",
                "width": "150px",
                "margin": "auto"
            }),
            html.Div("SNR", style={
                "marginTop": "10px",
                "fontSize": "20px",
                "color": "#1f2c56",
                "fontFamily": "Arial"
            })
        ], style=indicador_estilo)

        sdnr_box = html.Div([
            html.Div(f"{sdnr_val:.2f}", style={
                "border": "3px solid #007BFF",
                "backgroundColor": "#E6F0FF",
                "borderRadius": "15px",
                "padding": "30px",
                "fontSize": "50px",
                "color": "black",
                "fontFamily": "Arial Black",
                "width": "150px",
                "margin": "auto"
            }),
            html.Div("SDNR", style={
                "marginTop": "10px",
                "fontSize": "20px",
                "color": "#1f2c56",
                "fontFamily": "Arial"
            })
        ], style=indicador_estilo)

        snr_sdnr_panel = html.Div([snr_box, sdnr_box], style={"textAlign": "center"})

        mtf_h = row["MTF_H"]
        mtf_v = row["MTF_V"]

        if not mtf_h and not mtf_v:
            mtf_fig = go.Figure().update_layout(title="MTF no disponible")
        else:
            mtf_df = pd.DataFrame({
                "Frecuencia (ciclos/mm)": list(mtf_h.keys()) + list(mtf_v.keys()),
                "Valor MTF": list(mtf_h.values()) + list(mtf_v.values()),
                "Orientación": ["Horizontal"] * len(mtf_h) + ["Vertical"] * len(mtf_v)
            })
            mtf_fig = go.Figure()
            for orient in ["Horizontal", "Vertical"]:
                df_ = mtf_df[mtf_df["Orientación"] == orient]
                mtf_fig.add_trace(go.Scatter(
                    x=df_["Frecuencia (ciclos/mm)"],
                    y=df_["Valor MTF"],
                    mode='lines+markers',
                    name=orient
                ))
            mtf_fig.update_layout(title="Curva MTF")

    else:
        info_text = html.Div([html.H4("Tendencias Globales")])

        snr_sdnr_fig = px.line(
            df.sort_values("Fecha"),
            x="Fecha",
            y=["SNR", "SDNR"],
            markers=True,
            title="Evolución temporal de SNR y SDNR"
        )

        mtf50_data = []
        for _, row in df.iterrows():
            mtf_h = row["MTF_H"]
            mtf_v = row["MTF_V"]
            entry = {
                "Fecha": row["Fecha"],
                "UID": row["UID"],
                "MTF50_H": mtf_h.get("MTF@50%", None),
                "MTF50_V": mtf_v.get("MTF@50%", None)
            }
            if entry["MTF50_H"] is not None or entry["MTF50_V"] is not None:
                mtf50_data.append(entry)

        if mtf50_data:
            mtf50_df = pd.DataFrame(mtf50_data).sort_values("Fecha")
            mtf_fig = px.line(
                mtf50_df,
                x="Fecha",
                y=["MTF50_H", "MTF50_V"],
                markers=True,
                title="Evolución MTF@50%",
                labels={"value": "MTF@50%", "variable": "Orientación"},
                hover_data=["UID"]
            )
        else:
            mtf_fig = go.Figure().update_layout(title="MTF@50% no disponible")

        snr_sdnr_panel = html.Div([dcc.Graph(figure=snr_sdnr_fig)])

    return info_text, snr_sdnr_panel, mtf_fig

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001)
