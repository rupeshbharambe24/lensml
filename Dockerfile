# Fresh-environment install smoke test (plan §8 D29 → step 5.1).
#
# Build:  docker build -t lensml-smoke .
# Run:    docker run --rm lensml-smoke
#
# This image proves that a brand-new container can `pip install -e .`,
# import the package, and run `lensml --help` without any local cruft
# from the dev machine. The Unstop submission asks for a reproducible
# install path; this is it.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for libcst / numpy / torch wheel install.
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy only the install surface first to maximize Docker layer caching.
COPY pyproject.toml README.md LICENSE LICENSE-EE ./
COPY lensml ./lensml

# Editable install + verify import + show CLI help.
RUN pip install --upgrade pip \
 && pip install -e . \
 && python -c "import lensml; print('import OK:', lensml.__name__)" \
 && lensml --help

# Final image just runs the smoke verification on `docker run`.
COPY scripts/smoke_install.sh /usr/local/bin/smoke
RUN chmod +x /usr/local/bin/smoke

CMD ["smoke"]
