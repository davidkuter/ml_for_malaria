#!/bin/sh
pip-compile -o requirements/requirements.txt
pip-compile --extra=dev --extra=test -o requirements/requirements-dev.txt