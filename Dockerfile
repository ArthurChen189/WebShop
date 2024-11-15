FROM mambaorg/micromamba:debian-slim
USER root

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install --yes --file /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

RUN apt update
RUN apt install --yes build-essential wget
RUN apt clean --yes

WORKDIR /opt/webshop/data
COPY requirements.txt .
RUN pip install -r requirements.txt
# fix for flask not specifying dependency
RUN pip install Werkzeug==2.2.2
RUN pip install --force-reinstall typing-extensions==4.5.0
RUN pip install -U numpy
# Download spaCy large NLP model
RUN python -m spacy download en_core_web_lg

RUN wget https://s3.us-west-1.wasabisys.com/vzhong-public/webshop_data.tar.bz2
RUN tar -xvjf webshop_data.tar.bz2
RUN rm webshop_data.tar.bz2
RUN gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O

WORKDIR /opt/webshop
COPY . .
WORKDIR /opt/webshop/search_engine

RUN mkdir -p indexes resources resources_100 resources_1k resources_100k
# convert items.json => required doc format
RUN pip install ujson
RUN python convert_product_file_format.py
# Build search engine index
RUN ./run_indexing.sh
WORKDIR /opt/webshop/user_session_logs
RUN gdown https://drive.google.com/drive/u/1/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto

RUN python -m spacy download en_core_web_sm
WORKDIR /opt/webshop
