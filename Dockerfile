FROM python:3.12-slim AS build
WORKDIR /app
RUN pip install --no-cache-dir pyyaml pillow jsonschema
COPY . .
RUN python scripts/build-site.py

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/public /usr/share/nginx/html
