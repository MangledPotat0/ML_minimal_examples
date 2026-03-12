#!/bin/bash

source ../.env

thing=$(docker images | grep "mlplayground")
echo $thing

if [ -n "$thing" ]; then
    echo "Docker image \"mlplayground\" already exists."
else
    echo "Building Docker image \"geonosis\"..."
    docker build -t mlplayground .
fi
