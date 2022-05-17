from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import logging
import uvicorn

from . import mlib

logging.basicConfig(level=logging.INFO)

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
                    <a href="https://ml.stravasnooper.com/docs">ml.stravasnooper.com/docs</a></strong></p>"""
    return HTMLResponse(content=intro_message, status_code=200)


class PredictedKudos(BaseModel):
    perc_follwers: List[float]
    num_kudos: List[int]


@app.get("/kudos-prediction", response_model=PredictedKudos)
async def predict_kudos(
    custom_name: List[int] = Query(
        default=None,
        ge=0,
        le=1,
        description="Provide 0 for standard Strava name, 1 for custom name.",
    ),
    distance: List[int] = Query(
        default=None, ge=10, le=250, description="Ride distance in km's."
    ),
    achievements: List[int] = Query(
        default=None, ge=0, le=60, description="No. of achievements on the ride."
    ),
    elevation: List[int] = Query(
        default=None, ge=0, le=3000, description="Elevation gain in m's."
    ),
    followers: int = Query(
        default=None, ge=10, le=100000, description="Current number of followers."
    ),
):

    # check each pass has matching number of inputs
    if len(set(map(len, [custom_name, distance, achievements, elevation]))) != 1:
        raise HTTPException(
            status_code=404,
            detail="Number of values passed for parameters do not match.",
        )
    else:

        perc_followers, num_kudos = mlib.predict(
            custom_name_bool=custom_name,
            achievement_count=achievements,
            total_elevation_gain=elevation,
            distance_km=distance,
            num_followers=followers,
        )

        predict_result = {
            "perc_followers": perc_followers,
            "num_kudos": num_kudos,
        }

        logging.debug(f"Prediction: {predict_result}")
        print(predict_result)

        return predict_result


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")
