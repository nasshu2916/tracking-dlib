#! /bin/bash

echo "run format"
isort .
yapf -ri .
