import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, Input, Output, State
import io
import base64

# Cargar el dataset
df = pd.read_csv("./Pokemon_processed.csv")

# Características seleccionadas
features = ['Agresividad', 'Resistencia', 'Movilidad', 'Especialista ofensivo', 'Especialista defensivo', 'Balanceado']
X = df[features]

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Nombres de clanes por cluster
clan_names = {
    0: "Sombra",
    1: "Estratega",
    2: "Furia",
    3: "Coloso"
} 
df['Clan'] = df['Cluster'].map(clan_names)

# Reducción de dimensión para visualización
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'], df['PCA3'] = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]

# Calcular nombres para los ejes basados en las características
feature_weights = np.abs(pca.components_)
axis_names = []
for i in range(3):
    # Encontrar la característica con mayor peso en este componente
    top_feature_idx = np.argmax(feature_weights[i])
    axis_names.append(features[top_feature_idx])

# Calcular varianza explicada
variance_explained = pca.explained_variance_ratio_
total_variance = sum(variance_explained)

# Precalcular datos para visualizaciones
cluster_means = df.groupby('Clan')[features].mean().reset_index()
global_means = df[features].mean()
cluster_differences = cluster_means.copy()
for feature in features:
    cluster_differences[feature] = cluster_differences[feature] - global_means[feature]

# Inicializar app Dash
app = Dash(__name__)
server = app.server  # Si lo despliegas en la web
app = Dash(__name__, suppress_callback_exceptions=True)

# Layout de la app
app.layout = html.Div(
    style={
        'backgroundImage': 'url("/assets/pika.jpg")',
        'backgroundSize': 'cover',
        'backgroundRepeat': 'no-repeat',
        'backgroundPosition': 'center center',
        'minHeight': '100vh',
        'padding': '20px',
        'color': 'black'
    },
    children=[
        html.Audio(
            src="/assets/Pokemon.mp3", 
            controls=False,
            autoPlay=True,
            loop=True,
            style={"display": "none"}
        ),

        html.H1("Clustering de Pokémon con KMeans y PCA", 
                style={'textAlign': 'center', 'marginBottom': '40px'}),

        dcc.Tabs(id='tabs', value='tab-3d', style={'backgroundColor': 'rgba(224, 255, 98, 0.8)', 'border': 'none'},  children=[
            dcc.Tab(label='🏔️ Visualización 3D', value='tab-3d', style={'backgroundColor': 'rgba(247,254,212,0.8)'}),
            dcc.Tab(label='📊 Matriz de Dispersión', value='tab-pair', style={'backgroundColor': 'rgba(247,254,212,0.8)'}),
            dcc.Tab(label='📈 Perfiles de Clan', value='tab-radar', style={'backgroundColor': 'rgba(247,254,212,0.8)'}),
            dcc.Tab(label='🔥 Comparación de Clanes', value='tab-heatmap', style={'backgroundColor': 'rgba(247,254,212,0.8)'}),
            dcc.Tab(label='🧵 Proyección Paralela', value='tab-parallel', style={'backgroundColor': 'rgba(247,254,212,0.8)'}),
            dcc.Tab(label='🔍 Explorador de Pokémon', value='tab-explorer', style={'backgroundColor': 'rgba(247,254,212,0.8)'})
        ]),

        html.Div(id='tabs-content', style={'marginTop': 20}),
    
    ])
# Callback para actualizar contenido de pestañas
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_tab(tab):
    if tab == 'tab-pair':
        # Crear la figura de matriz de dispersión
        fig = px.scatter_matrix(
            df,
            dimensions=features,
            color="Clan",
            hover_name="Name",
            title="Matriz de Dispersión: Relaciones entre características"
        )
        
        # Ajustes para mejorar la legibilidad
        fig.update_layout(
            height=1000,
            width=1200,
            font=dict(size=10)
        )
        
        # Rotar las etiquetas de los ejes y ajustar espaciado
        fig.update_xaxes(tickangle=45, tickfont=dict(size=9))
        fig.update_yaxes(tickangle=-45, tickfont=dict(size=9))
        
        # Ajustar el espaciado entre subplots
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(
            margin=dict(l=50, r=50, b=50, t=80),
            showlegend=True
        )
        
        return html.Div([
            dcc.Graph(figure=fig),
            html.P("🔍 Consejo: Usa el zoom (rueda del mouse) y arrastra para explorar relaciones específicas"),
            html.P("💡 Las celdas en la diagonal muestran la distribución de cada característica por clan")
        ], style={'overflowX': 'auto'})
    
    elif tab == 'tab-radar':
        # Crear figura de radar
        radar_fig = go.Figure()
        for i, clan in enumerate(cluster_means['Clan']):
            clan_data = cluster_means[cluster_means['Clan'] == clan]
            radar_fig.add_trace(go.Scatterpolar(
                r=clan_data[features].values[0],
                theta=features,
                fill='toself',
                name=clan,
                line_color=px.colors.qualitative.Plotly[i]
            ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 0.4])),
            showlegend=True,
            title="Perfil promedio de cada Clan",
            height=500
        )
        return html.Div([
            dcc.Graph(figure=radar_fig),
            html.P("Interpretación: Este gráfico muestra el perfil promedio de cada clan. Permite identificar fortalezas y debilidades específicas de cada grupo.")
        ])
    
    elif tab == 'tab-heatmap':
        # Crear mapa de calor
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            cluster_differences.set_index('Clan')[features],
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5
        )
        plt.title("Desviación de características respecto a la media global por Clan")
        plt.tight_layout()
        
        # Convertir a imagen
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        heatmap_image = base64.b64encode(buf.read()).decode('utf-8')
        return html.Div([
            html.Img(src=f'data:image/png;base64,{heatmap_image}'),
            html.P("Interpretación: Muestra cómo cada clan se desvía de la media global en cada característica.")
        ])
    
    elif tab == 'tab-parallel':
        return html.Div([
            dcc.Graph(
                figure=px.parallel_coordinates(
                    df,
                    color="Cluster",
                    dimensions=features,
                    labels={"Cluster": "Clan"},
                    color_continuous_scale=px.colors.diverging.Tealrose,
                    title="Proyección Paralela de Características"
                )
            ),
            html.P("Interpretación: Cada línea representa un Pokémon. Permite identificar patrones y valores atípicos.")
        ])
    
    elif tab == 'tab-explorer':
        return html.Div([
            html.H3("Explorador de Pokémon por Clan"),
            dcc.Dropdown(
                id='clan-dropdown',
                options=[{'label': clan, 'value': clan} for clan in df['Clan'].unique()],
                value='Furia',
                style={'width': '50%', 'marginBottom': '20px'}
            ),
            html.Div(id='clan-stats'),
            dcc.Graph(id='clan-scatter')
        ])
    
    else:  # tab-3d por defecto
        return html.Div([
            # Columna izquierda: Dropdown + info
            html.Div([
                dcc.Dropdown(
                    id='pokemon-dropdown',
                    options=[{'label': name, 'value': name} for name in df['Name'].sort_values()],
                    placeholder="Selecciona un Pokémon",
                    style={'width': '100%', 'marginBottom': '20px'}
                ),
                html.Div(id='pokemon-info', style={'fontSize': 16})
            ], style={
                'width': '35%',
                'padding': '20px',
                #'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                'borderRadius': '10px',
                'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.1)',
                'marginRight': '20px'
            }),

            # Columna derecha: Gráfico 3D
            html.Div([
                dcc.Graph(
                    id='scatter-3d',
                    config={
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'editable': False,
                        'staticPlot': False
                    },
                    style={'height': '600px'}
                )
            ], style={'width': '65%'})
        ], style={'display': 'flex', 'justifyContent': 'space-between'})


# Callback para actualizar gráfico e info
@app.callback(
    Output('scatter-3d', 'figure'),
    Output('pokemon-info', 'children'),
    Input('pokemon-dropdown', 'value')
)
def actualizar_grafico_y_info(nombre):
    # Si hay Pokémon seleccionado, mostrarlo destacado
    if nombre:
        poke = df[df['Name'] == nombre].iloc[0]

        # Todos los puntos con opacidad baja
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=df['PCA1'],
            y=df['PCA2'],
            z=df['PCA3'],
            mode='markers',
            marker=dict(
                size=4,
                color=df['Cluster'],  # usa el color del cluster
                opacity=0.7
            ),
            text=df['Name'],
            hoverinfo='text',
            name='Todos los Pokémon'
        ))

        # Pokémon seleccionado resaltado
        fig.add_trace(go.Scatter3d(
            x=[poke['PCA1']],
            y=[poke['PCA2']],
            z=[poke['PCA3']],
            mode='markers+text',
            marker=dict(
                size=10,
                color= 'black',
                line=dict(color='white', width=3),
                opacity=0.8
            ),
            hovertemplate=(
                f"<b>{poke['Name']}</b><br>"
                f"Clan: {poke['Clan']}<br>"
                f"Agresividad: {poke['Agresividad']:.2f}<br>"
                f"Resistencia: {poke['Resistencia']:.2f}<br>"
                f"Movilidad: {poke['Movilidad']:.2f}<br>"
                f"Especialista ofensivo: {poke['Especialista ofensivo']:.2f}<br>"
                f"Especialista defensivo: {poke['Especialista defensivo']:.2f}<br>"
                f"Balanceado: {poke['Balanceado']:.2f}<extra></extra>"
            ),
            text=[poke['Name']],
            textposition="top center",
            name=f"{poke['Name']} (seleccionado)",
            showlegend=True
        ))

        # Texto informativo
        info = html.Div([
            html.H4(poke['Name'], style={'margin-bottom': '0.2em'}),
            html.H5(f"Clan: {poke['Clan']}", style={'margin-top': '0'}),

            html.Hr(style={'margin': '0.5em 0'}),

            html.Div([
                html.Div([
                    html.Strong("Agresividad: "),
                    f"{poke['Agresividad']:.2f}"
                ], style={'margin-bottom': '0.2em'}),
        
                html.Div([
                    html.Strong("Resistencia: "),
                    f"{poke['Resistencia']:.2f}"
                ], style={'margin-bottom': '0.2em'}),
        
                html.Div([
                    html.Strong("Movilidad: "),
                    f"{poke['Movilidad']:.2f}"
                    ], style={'margin-bottom': '0.2em'}),
                ], style={'margin-bottom': '0.5em'}),

            html.Div([
                html.Div([
                    html.Strong("Especialista ofensivo: "),
                    f"{poke['Especialista ofensivo']:.2f}"
                ], style={'margin-bottom': '0.2em'}),

                html.Div([
                    html.Strong("Especialista defensivo: "),
                    f"{poke['Especialista defensivo']:.2f}"
                ], style={'margin-bottom': '0.2em'}),

                html.Div([
                    html.Strong("Balanceado: "),
                    f"{poke['Balanceado']:.2f}"
                ])
            ])
        ], style={
            'border': '2px solid #ccc',
            'border-radius': '10px',
            'padding': '15px',
            'background-color': "#ffffff",
            'width': '300px',
            'font-family': 'Arial, sans-serif',
            'box-shadow': '2px 2px 10px rgba(0, 0, 0, 0.1)',
            'margin-top': '20px',
}
)
    else:
        # Si no hay selección, mostrar todos con opacidad normal
        fig = px.scatter_3d(
            df, x='PCA1', y='PCA2', z='PCA3',
            color='Clan',
            hover_name='Name',
            hover_data={
                        'Agresividad': ':.2f',
                        'Resistencia': ':.2f',
                        'Movilidad': ':.2f',
                        'Especialista ofensivo': ':.2f',
                        'Especialista defensivo': ':.2f',
                        'Balanceado': ':.2f',
                        'PCA1': False,
                        'PCA2': False,
                        'PCA3': False,
                        'Cluster': False
            },
            labels={'PCA1': 'Componente principal 1', 'PCA2': 'Componente principal 2', 'PCA3': 'Componente principal 3'},
            opacity=1
        )
        info = ""

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.01, y=0.99, font=dict(size=15, color='black')),
        scene=dict(
            xaxis=dict(backgroundcolor='rgba(255,255,84,0.8)', showbackground=True, color='black', gridcolor='black'),
            yaxis=dict(backgroundcolor='rgba(209,255,84,0.8)', showbackground=True, color='black', gridcolor='black'),
            zaxis=dict(backgroundcolor='rgba(209,255,84,0.8)', showbackground=True, color='black', gridcolor='black')
        ),
        paper_bgcolor='rgba(255,255,255,0)',  # fondo del gráfico
        plot_bgcolor='rgba(0,0,0,0)'    # fondo del área de trazado
    )

    return fig, info

# Callback para explorador de clanes
@app.callback(
    Output('clan-stats', 'children'),
    Output('clan-scatter', 'figure'),
    Input('clan-dropdown', 'value')
)
def actualizar_clan_info(clan):
    clan_df = df[df['Clan'] == clan]
    
    # Estadísticas principales
    stats_html = html.Div([
        html.H4(f"Estadísticas del Clan: {clan}"),
        html.P(f"Número de Pokémon: {len(clan_df)}"),
        html.P(f"Agresividad promedio: {clan_df['Agresividad'].mean():.3f}"),
        html.P(f"Resistencia promedio: {clan_df['Resistencia'].mean():.3f}"),
        html.P(f"Movilidad promedio: {clan_df['Movilidad'].mean():.3f}"),
        html.H5("Tipos más comunes:"),
        html.Ul([html.Li(tipo) for tipo in clan_df['Type 1'].value_counts().head(3).index])
    ])
    
    # Gráfico de dispersión para el clan
    scatter_fig = px.scatter(
        clan_df,
        x='Agresividad',
        y='Resistencia',
        color='Type 1',
        hover_name='Name',
        size='Total',
        title=f"Distribución de Pokémon en el Clan {clan}",
        labels={'Agresividad': 'Agresividad', 'Resistencia': 'Resistencia'}
    )
    
    return stats_html, scatter_fig


# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True)
