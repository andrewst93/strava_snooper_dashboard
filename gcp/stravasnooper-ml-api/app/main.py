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
    intro_message = """<h2 style="text-align: center;"><img src="https://www.stravasnooper.com/assets/images/strava_snooper_wide_logo.png" alt="" width="636" height="82" /></h2>
        <h2 style="text-align: center;">Welcome To The StravaSnooper ML API</h2>
        <p style="text-align: center;">These API's look to help us better understand our Strava data and what we can do with it. Models are developed and hosted by Ty Andrews.</p>
        <p style="text-align: center;"><a href="/docs">API Documentation</a></p>
        <h3 style="text-align: center;">&nbsp;<strong>Kudos Prediction API</strong></h3>
        <p style="text-align: center;">Predict how many kudos you'll get based on your upcoming ride plan and you can try to maximize them.</p>
        <p style="text-align: center;"><a href="https://www.stravasnooper.com/pages/kudos-prediction" target="_blank" rel="noopener">Try the Demo Here</a></p>"""
    return HTMLResponse(content=intro_message, status_code=200)


class PredictedKudos(BaseModel):
    perc_followers: List[float]
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
        default=None, ge=0, le=5000, description="Elevation gain in m's."
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
