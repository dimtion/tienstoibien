#!/usr/bin/env bash

ENV=env
REQUIREMENTS=requirements.txt

if [ ! -d "$ENV" ]; then
        python3 -m virtualenv "$ENV"
        "$ENV/bin/pip" install -Ur "$REQUIREMENTS"
fi

# Startup
"$ENV/bin/python" main.py
