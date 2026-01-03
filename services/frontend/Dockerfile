FROM node:18-alpine
WORKDIR /app
COPY . /app
RUN npm install # only if package.json exists; otherwise no-op
# If you have a dev server start script like `npm start`
CMD ["npx", "http-server", "-p", "3000", "./"]
