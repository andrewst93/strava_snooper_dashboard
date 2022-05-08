FROM python:3.7-bullseye

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

COPY ./models /code/models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]



# FROM ubuntu:latest
# LABEL author="Ty Andrews"
# LABEL version="0.0.1"

# #from here to try and resolve no_pubkeys error on build: https://stackoverflow.com/questions/63472872/building-from-dockerfile-and-getting-the-following-signatures-couldnt-be-verif
# # RUN for key in 871920D1991BC93C; do \
# #     gpg --recv-keys "$key" \
# #     && gpg --export "$key" | apt-key add - ; \
# #     done

# RUN apt-get update \  
#     && apt-get install -y python3-pip python3-dev \  
#     && cd /usr/local/bin \  
#     && ln -s /usr/bin/python3 python \  
#     && pip3 install flask  

# COPY kudos-prediction-endpoint.py kudos-prediction-endpoint.py 
# ENTRYPOINT ["python3", "kudos-prediction-endpoint.py"]