# syntax=docker/dockerfile:1
FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM jonasal/nginx-certbot:6.2.0
ARG DOMAIN=localhost
COPY --from=build /app/dist/frontend/browser /usr/share/nginx/html
COPY nginx/user_conf.d/ /etc/nginx/user_conf.d/
COPY nginx/proxy_common.conf /etc/nginx/proxy_common.conf
RUN sed -i "s|__DOMAIN__|${DOMAIN}|g" /etc/nginx/user_conf.d/dsnavigator.conf
