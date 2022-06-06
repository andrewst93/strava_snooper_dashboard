import requests


def predict_kudos(custom_name, distance, achievements, elevation, num_followers):
    """Calls the StravaSnooper API to get predictions of kudos from ride data

    Args:
        custom_name (List[int:0-1]): Indicates if standard strava name or custom name. 0 for standard, 1 for custom name.
        distance (List[int]): Ride distance in km's.
        achievements (List[int]): No. of achievements on the ride  .
        elevation (List[int]): Elevation gain over ride in m's.
        num_followers (int): How many followers the user has.

    Returns:
        num_kudos       (List[int]): The predicted number of kudos.
                                        Returns None if issue with API request.
        perc_followers  (List[float]): What percentage of users followers will give kudos.
                                        Returns None if issue with API request.
    """

    data = {
        "custom_name": custom_name,
        "distance": distance,
        "achievements": achievements,
        "elevation": elevation,
        "followers": num_followers,
    }

    resp = requests.get("https://ml.stravasnooper.com/kudos-prediction", params=data)

    # successful request
    if resp.status_code == 200:
        num_kudos = resp.json()["num_kudos"]
        perc_followers = resp.json()["perc_followers"]
        return num_kudos, perc_followers
    else:
        print(
            f"[predict_kudos] Failed API request, url: {resp.url}, response: {resp.text}"
        )
        return None, None
