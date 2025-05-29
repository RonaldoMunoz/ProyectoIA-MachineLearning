import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

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

# Inicializar app Dash
app = Dash(__name__)
server = app.server  # Si lo despliegas en la web

# Layout de la app
app.layout = html.Div(
    style={
        'backgroundImage': 'url("/assets/pokemon.jpeg")',
        'backgroundSize': 'cover',
        'backgroundRepeat': 'no-repeat',
        'backgroundPosition': 'center center',
        'minHeight': '100vh',
        'padding': '20px',
        'position': 'relative'
    },
    children=[
        html.Audio(
            src="/assets/Pokemon.mp3", 
            controls=False,
            autoPlay=True,
            loop=True,
            style={"display": "none"}  # Oculta el reproductor
        ),

        html.H1("Clustering de Pokémon con KMeans y PCA", style={'textAlign': 'center', 'marginBottom': '20px'}),
    
        dcc.Dropdown(
            id='pokemon-dropdown',
            options=[{'label': name, 'value': name} for name in df['Name'].sort_values()],
            placeholder="Selecciona un Pokémon",
            style={'width': '50%'}
        ),

        html.Div(id='pokemon-info', style={'marginTop': 20, 'fontSize': 16}),
    
        dcc.Graph(
            id='scatter-3d',
            style={
                'height': '600px', 
                'width': '50%',
                'float': 'right',
                'margin': 'auto',
                'marginRight': '40px',
                'marginTop':'-85px'},
                
            config={
            'scrollZoom': True,     
            'displayModeBar': True, 
            'editable': False,      
            'staticPlot': False,     
    })
    ]
)

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
            'background-color': '#f9f9f9',
            'width': '300px',
            'font-family': 'Arial, sans-serif',
            'box-shadow': '2px 2px 10px rgba(0, 0, 0, 0.1)',
            'margin-top': '20px',
        })

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
        legend=dict(x=0.01, y=0.99, font=dict(size=15, color='white')),
        scene=dict(
            xaxis=dict(backgroundcolor='rgba(209,255,255,1)', showbackground=True, color='white', gridcolor='black'),
            yaxis=dict(backgroundcolor='rgba(209,255,255,1)', showbackground=True, color='white', gridcolor='black'),
            zaxis=dict(backgroundcolor='rgba(209,255,255,1)', showbackground=True, color='white', gridcolor='black')
        ),
        paper_bgcolor='rgba(255,255,255,0)',  # fondo del gráfico
        plot_bgcolor='rgba(0,0,0,0)'    # fondo del área de trazado
    )

    return fig, info

# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True)
