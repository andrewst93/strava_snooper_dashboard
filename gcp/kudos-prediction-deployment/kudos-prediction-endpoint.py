# load Flask
import flask

app = flask.Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/query-example")
def query_example():
    # if key doesn't exist, returns None
    language = flask.request.args.get("language")

    return """<h1>The language value is: {}</h1>""".format(language)


# define a predict function as an endpoint
@app.route("/predict-kudos", methods=["GET"])
def predict_kudos():
    result = {"success": False}
    # get the request parameters
    data = flask.request.args

    if "distance" in data:
        distance = data["distance"]
    if "elevation" in data:
        elevation = data["elevation"]
    if "achievements" in data:
        achievements = data["achievements"]
    if "ride-name" in data:
        ride_name = data["ride-name"]

    # if all parameters are found and not none, echo the msg parameter
    if [distance, elevation, achievements, ride_name].count(None) == 0:
        result[
            "response"
        ] = f"Kudos from: {distance}km, {elevation}m, {ride_name} ride name, {achievements}ach"
        result["success"] = True
    # return a response in json format
    return flask.jsonify(result)


# start the flask app, allow remote connections
app.run(host="0.0.0.0", port=8090)
