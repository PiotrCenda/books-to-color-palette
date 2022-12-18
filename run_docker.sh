#!/bin/bash

docker build -t test .
docker run --name test_con -v /results:/usr/test/data/gifs --rm test
