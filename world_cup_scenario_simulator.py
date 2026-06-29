import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# =====================================================
# WORLD CUP SCENARIO SIMULATOR
# Machine Learning + Monte Carlo
# =====================================================

np.random.seed(42)

print("=" * 70)
print("🌍 WORLD CUP SCENARIO SIMULATOR")
print("Machine Learning + Monte Carlo Simulation")
print("=" * 70)


# =====================================================
# 1. CARGAR DATOS
# =====================================================

print("\n📂 Cargando datos...")

matches = pd.read_csv("matches.csv")
teams = pd.read_csv("teams.csv")
players = pd.read_csv("squads_and_players_ascii_mysql_ready.csv")
venues = pd.read_csv("venues.csv")
stages = pd.read_csv("tournament_stages.csv")

print(f"✔ Partidos cargados: {matches.shape[0]}")
print(f"✔ Selecciones cargadas: {teams.shape[0]}")
print(f"✔ Jugadores cargados: {players.shape[0]}")


# =====================================================
# 2. CREAR VARIABLES DE EQUIPO
# =====================================================

print("\n🧹 Preparando datos...")

team_players = players.groupby("team_id").agg(
    squad_market_value=("market_value_eur", "sum"),
    avg_player_value=("market_value_eur", "mean"),
    avg_caps=("caps", "mean"),
    total_caps=("caps", "sum"),
    avg_height=("height_cm", "mean"),
    total_player_goals=("goals", "sum")
).reset_index()

team_data = teams.merge(team_players, on="team_id", how="left").fillna(0)


# =====================================================
# 3. CREAR DATASET DEL MODELO
# =====================================================

df = matches.copy()

df = df.merge(
    team_data,
    left_on="home_team_id",
    right_on="team_id",
    how="left"
)

df = df.rename(columns={
    "team_name": "home_team",
    "fifa_ranking_pre_tournament": "home_ranking",
    "elo_rating": "home_elo",
    "squad_market_value": "home_market_value",
    "avg_player_value": "home_avg_player_value",
    "avg_caps": "home_avg_caps",
    "total_caps": "home_total_caps",
    "avg_height": "home_avg_height",
    "total_player_goals": "home_total_goals"
})

df = df.drop(columns=[
    "team_id", "fifa_code", "group_letter", "confederation", "manager_name"
])


df = df.merge(
    team_data,
    left_on="away_team_id",
    right_on="team_id",
    how="left"
)

df = df.rename(columns={
    "team_name": "away_team",
    "fifa_ranking_pre_tournament": "away_ranking",
    "elo_rating": "away_elo",
    "squad_market_value": "away_market_value",
    "avg_player_value": "away_avg_player_value",
    "avg_caps": "away_avg_caps",
    "total_caps": "away_total_caps",
    "avg_height": "away_avg_height",
    "total_player_goals": "away_total_goals"
})

df = df.drop(columns=[
    "team_id", "fifa_code", "group_letter", "confederation", "manager_name"
])

df = df.merge(venues, on="venue_id", how="left")
df = df.merge(stages, on="stage_id", how="left")


# =====================================================
# 4. VARIABLE OBJETIVO
# 0 = gana local
# 1 = empate
# 2 = gana visitante
# =====================================================

def get_result(row):
    if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
        return np.nan

    if row["home_score"] > row["away_score"]:
        return 0
    elif row["home_score"] == row["away_score"]:
        return 1
    else:
        return 2


df["result"] = df.apply(get_result, axis=1)


# =====================================================
# 5. FEATURE ENGINEERING
# =====================================================

df["ranking_diff"] = df["home_ranking"] - df["away_ranking"]
df["elo_diff"] = df["home_elo"] - df["away_elo"]
df["market_value_diff"] = df["home_market_value"] - df["away_market_value"]
df["avg_player_value_diff"] = df["home_avg_player_value"] - df["away_avg_player_value"]
df["avg_caps_diff"] = df["home_avg_caps"] - df["away_avg_caps"]
df["total_caps_diff"] = df["home_total_caps"] - df["away_total_caps"]
df["height_diff"] = df["home_avg_height"] - df["away_avg_height"]
df["player_goals_diff"] = df["home_total_goals"] - df["away_total_goals"]

features = [
    "ranking_diff",
    "elo_diff",
    "market_value_diff",
    "avg_player_value_diff",
    "avg_caps_diff",
    "total_caps_diff",
    "height_diff",
    "player_goals_diff",
    "elevation_meters",
    "capacity",
    "is_knockout"
]


# =====================================================
# 6. ENTRENAR MODELOS
# =====================================================

played = df[df["result"].notna()].copy()

X = played[features].fillna(0)
y = played["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

best_model = None
best_name = ""
best_accuracy = 0

print("\n🧠 Entrenando modelos...")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(f"✔ {name:<22} {acc * 100:.2f}%")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

print("\n🏆 Mejor modelo:", best_name)
print(f"🎯 Accuracy: {best_accuracy * 100:.2f}%")


# =====================================================
# 7. FUNCIONES DE PREDICCIÓN
# =====================================================

team_dict = team_data.set_index("team_id").to_dict("index")


def build_match_features(home_id, away_id, is_knockout=0):
    home = team_dict[home_id]
    away = team_dict[away_id]

    return pd.DataFrame([{
        "ranking_diff": home["fifa_ranking_pre_tournament"] - away["fifa_ranking_pre_tournament"],
        "elo_diff": home["elo_rating"] - away["elo_rating"],
        "market_value_diff": home["squad_market_value"] - away["squad_market_value"],
        "avg_player_value_diff": home["avg_player_value"] - away["avg_player_value"],
        "avg_caps_diff": home["avg_caps"] - away["avg_caps"],
        "total_caps_diff": home["total_caps"] - away["total_caps"],
        "height_diff": home["avg_height"] - away["avg_height"],
        "player_goals_diff": home["total_player_goals"] - away["total_player_goals"],
        "elevation_meters": 0,
        "capacity": 0,
        "is_knockout": is_knockout
    }])


def predict_match_probabilities(home_id, away_id, is_knockout=0):
    row = build_match_features(home_id, away_id, is_knockout)
    probs = best_model.predict_proba(row)[0]
    classes = best_model.classes_

    prob_dict = dict(zip(classes, probs))

    home_win = prob_dict.get(0, 0)
    draw = prob_dict.get(1, 0)
    away_win = prob_dict.get(2, 0)

    total = home_win + draw + away_win

    if total == 0:
        return 1/3, 1/3, 1/3

    return home_win / total, draw / total, away_win / total


def predict_knockout_winner(team_a_id, team_b_id):
    home_win, draw, away_win = predict_match_probabilities(
        team_a_id,
        team_b_id,
        is_knockout=1
    )

    # En eliminatorias no hay empate final.
    # Se redistribuye la probabilidad del empate.
    prob_a = home_win + (draw / 2)
    prob_b = away_win + (draw / 2)

    # Suavizar probabilidades para evitar resultados exagerados.
    prob_a = max(0.10, min(0.90, prob_a))
    prob_b = 1 - prob_a

    winner = np.random.choice(
        [team_a_id, team_b_id],
        p=[prob_a, prob_b]
    )

    return winner


# =====================================================
# 8. PREDICCIONES POR PARTIDO PENDIENTE
# =====================================================

pending = df[df["result"].isna()].copy()

pred_rows = []

print("\n⚽ Predicciones de partidos pendientes")
print("=" * 70)

for _, match in pending.iterrows():
    home_id = match["home_team_id"]
    away_id = match["away_team_id"]

    home_prob, draw_prob, away_prob = predict_match_probabilities(
        home_id,
        away_id,
        is_knockout=match["is_knockout"]
    )

    probs = [home_prob, draw_prob, away_prob]
    prediction = np.argmax(probs)

    if prediction == 0:
        predicted_result = f"Gana {match['home_team']}"
    elif prediction == 1:
        predicted_result = "Empate"
    else:
        predicted_result = f"Gana {match['away_team']}"

    print(f"\n{match['home_team']} vs {match['away_team']}")
    print(f"Gana {match['home_team']}: {home_prob * 100:.2f}%")
    print(f"Empate: {draw_prob * 100:.2f}%")
    print(f"Gana {match['away_team']}: {away_prob * 100:.2f}%")
    print(f"Predicción: {predicted_result}")

    pred_rows.append({
        "match_id": match["match_id"],
        "date": match["date"],
        "stage_name": match["stage_name"],
        "home_team": match["home_team"],
        "away_team": match["away_team"],
        "home_win_probability": round(home_prob * 100, 2),
        "draw_probability": round(draw_prob * 100, 2),
        "away_win_probability": round(away_prob * 100, 2),
        "predicted_result": predicted_result
    })

match_predictions = pd.DataFrame(pred_rows)


# =====================================================
# 9. SIMULACIÓN SIMPLE DEL CAMPEÓN
# =====================================================

print("\n🎲 Simulando escenarios de campeón...")

team_ids = teams["team_id"].tolist()
team_names = teams.set_index("team_id")["team_name"].to_dict()

# Tomamos como candidatos principales a los equipos con mejor Elo.
top_candidates = (
    teams.sort_values("elo_rating", ascending=False)
    .head(16)["team_id"]
    .tolist()
)

n_simulations = 3000
champions = []

for i in range(n_simulations):
    alive = top_candidates.copy()
    np.random.shuffle(alive)

    while len(alive) > 1:
        next_round = []

        for j in range(0, len(alive), 2):
            team_a = alive[j]
            team_b = alive[j + 1]

            winner = predict_knockout_winner(team_a, team_b)
            next_round.append(winner)

        alive = next_round

    champions.append(team_names[alive[0]])

    if (i + 1) % 500 == 0:
        print(f"✔ Simulación {i + 1}/{n_simulations}")


champion_probs = (
    pd.Series(champions)
    .value_counts(normalize=True)
    .reset_index()
)

champion_probs.columns = ["team", "champion_probability"]
champion_probs["champion_probability_pct"] = (
    champion_probs["champion_probability"] * 100
).round(2)


# =====================================================
# 10. MOSTRAR RESULTADO FINAL
# =====================================================

print("\n" + "=" * 70)
print("🏆 ESCENARIOS DE CAMPEÓN MÁS PROBABLES")
print("=" * 70)

top_10 = champion_probs.head(10).reset_index(drop=True)

for i, row in top_10.iterrows():
    bar = "█" * int(row["champion_probability_pct"] / 2)

    print(
        f"{i + 1:>2}. {row['team']:<18} "
        f"{row['champion_probability_pct']:>6.2f}% "
        f"{bar}"
    )

winner = champion_probs.iloc[0]

print("=" * 70)
print("\n🏆 Equipo con mayor probabilidad en esta simulación:")
print(f"Selección    : {winner['team']}")
print(f"Probabilidad : {winner['champion_probability_pct']}%")


# =====================================================
# 11. GUARDAR ARCHIVOS
# =====================================================

match_predictions.to_csv(
    "match_predictions_scenario_simulator.csv",
    index=False
)

champion_probs.to_csv(
    "champion_scenarios_simulator.csv",
    index=False
)

df.to_csv(
    "world_cup_scenario_dataset.csv",
    index=False
)

print("\n📁 Archivos generados:")
print("- match_predictions_scenario_simulator.csv")
print("- champion_scenarios_simulator.csv")
print("- world_cup_scenario_dataset.csv")

print("\n✅ Proceso finalizado correctamente.")