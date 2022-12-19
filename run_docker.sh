#!/bin/bash

echo "Creating docker image"
docker build -t test .

echo "Running container"
docker run --name test_con -v $(pwd)/results:/usr/test/data/gifs test
