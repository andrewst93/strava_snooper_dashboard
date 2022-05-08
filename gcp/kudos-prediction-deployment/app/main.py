from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from . import mlib

description = """
Stravasnooper API Helps You Understand Your Strava Data! 🚀

## Kudos Prediction

You can predict how many kudos you'll get on your next ride and see how to maximize your kudos.

For a live demo visit: https://www.stravasnooper.com/pages/kudos-prediction
"""

app = FastAPI(
    title="StravaSnooper API",
    description=description,
    version="0.0.1",
    contact={
        "name": "Ty Andrews",
        "url": "http://www.ty-andrews.com",
        "email": "ty.elgin.andrews@gmail.com",
    },
)


@app.get("/")
async def root():
    intro_message = """<h1><span style="color: #ff6600;">Welcome to the Stravasnooper ML API</span></h1> 
                    <p>This is a collection of models/tools developed from a completely unnecessary exploration of my Strava data.</p> 
                    <p><strong>To learn how to use this API for your own projects or interest visit: 
                    <a href="https://ml.stravasnooper.com/">ml.stravasnooper.com/docs</a></strong></p>"""
    return HTMLResponse(content=intro_message, status_code=200)


@app.get("/kudos-prediction")
async def predict_kudos(
    custom_name: int,
    distance: int,
    achievements: int,
    elevation: int,
    followers: int,
):

    prediction = mlib.predict(
        custom_name_bool=custom_name,
        achievement_count=achievements,
        total_elevation_gain=elevation,
        distance_km=distance,
        num_followers=followers,
    )

    return prediction


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")
