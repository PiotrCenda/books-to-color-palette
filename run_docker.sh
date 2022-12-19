#!/bin/bash

docker build -t test .
docker run --name test_con -v $(pwd)/results:/usr/test/data/gifs test
