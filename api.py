from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("match_predictor_model.pkl")

data = pd.read_csv("matches.csv")
data['date'] = pd.to_datetime(data['date'])

def get_team_features(team, match_date, data, max_matches=20):
    team_matches = data[((data['team'] == team) | (data['opponent'] == team)) & 
                        (data['date'] < match_date)].sort_values(by='date', ascending=False).head(max_matches)
    
    if len(team_matches) == 0:
        return [0] * 11 
    
    features = {}
    for col in ['shots', 'shots_on_target', 'possession', 'passes', 'pass_accuracy', 
                'fouls', 'yellow_cards', 'red_cards', 'offsides', 'corners']:
        features[f'avg_{col}'] = team_matches.apply(
            lambda row: row[col] if row['team'] == team else row[col], axis=1).mean()
    
    wins = len(team_matches[team_matches['result'] == 1])
    features['win_rate'] = wins / len(team_matches) if len(team_matches) > 0 else 0
    return list(features.values())

def predict_match_probabilities(team, opponent, match_date):
    match_date = pd.to_datetime(match_date)
    
    team_features = get_team_features(team, match_date, data)
    opponent_features = get_team_features(opponent, match_date, data)
    
    features = np.array([team_features + opponent_features])
    
    probabilities = model.predict_proba(features)[0]
    
    prob_dict = dict(zip(model.classes_, probabilities))
    
    win_prob = prob_dict.get(1, 0.0) * 100 
    draw_prob = prob_dict.get(0, 0.0) * 100 
    loss_prob = prob_dict.get(-1, 0.0) * 100 
    
    return {
        "win_probability": round(win_prob, 2),  
        "draw_probability": round(draw_prob, 2),  
        "loss_probability": round(loss_prob, 2)  
    }

@app.get("/predict")
def predict(team: str, opponent: str, date: str):
    try:
        probabilities = predict_match_probabilities(team, opponent, date)
        return probabilities
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطأ: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)