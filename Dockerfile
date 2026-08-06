FROM python:3.12-slim AS build
WORKDIR /app
# Pinned. The build is a validation gate, so its validator is load-bearing: an
# unpinned jsonschema that changes draft handling turns "the registry is wrong"
# into "the deploy is broken" with no commit to blame it on.
RUN pip install --no-cache-dir 'pyyaml~=6.0' 'pillow~=10.2' 'jsonschema~=4.10'
COPY . .
RUN python scripts/build-site.py

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/public /usr/share/nginx/html
