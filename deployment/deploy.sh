#!/bin/bash

cmd=$1

# constants
IMAGE_NAME="model_predictor"
IMAGE_TAG=$(git describe --always)

if [[ -z "$cmd" ]]; then
    echo "Missing command"
    exit 1
fi

run_predictor() {
    model_config_path=$1
    model_config_path2=$2
    if [[ -z "$model_config_path" ]]; then
        echo "Missing model_config_path"
        exit 1
    fi

    docker build -f deployment/model_predictor/Dockerfile -t $IMAGE_NAME:$IMAGE_TAG .
    IMAGE_NAME=$IMAGE_NAME IMAGE_TAG=$IMAGE_TAG \
        MODEL_CONFIG_PATH=$model_config_path MODEL_CONFIG_PATH2=$model_config_path2 \
        docker-compose -f deployment/model_predictor/docker-compose.yml up -d 
}

shift

case $cmd in
run_predictor)
    run_predictor "$@"
    ;;
*)
    echo -n "Unknown command: $cmd"
    exit 1
    ;;
esac