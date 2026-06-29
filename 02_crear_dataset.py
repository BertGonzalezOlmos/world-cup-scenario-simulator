import pandas as pd

# ==========================
# Cargar datos
# ==========================

matches = pd.read_csv("matches.csv")
teams = pd.read_csv("teams.csv")

# ==========================
# Agregar información del equipo local
# ==========================

matches = matches.merge(
    teams,
    left_on="home_team_id",
    right_on="team_id",
    how="left"
)

matches = matches.rename(columns={
    "team_name":"home_team",
    "elo_rating":"home_elo",
    "fifa_ranking_pre_tournament":"home_ranking"
})

# Eliminar columnas repetidas
matches = matches.drop(columns=[
    "team_id",
    "fifa_code",
    "group_letter",
    "confederation",
    "manager_name"
])

# ==========================
# Agregar información del visitante
# ==========================

matches = matches.merge(
    teams,
    left_on="away_team_id",
    right_on="team_id",
    how="left"
)

matches = matches.rename(columns={
    "team_name":"away_team",
    "elo_rating":"away_elo",
    "fifa_ranking_pre_tournament":"away_ranking"
})

matches = matches.drop(columns=[
    "team_id",
    "fifa_code",
    "group_letter",
    "confederation",
    "manager_name"
])

print(matches.head())