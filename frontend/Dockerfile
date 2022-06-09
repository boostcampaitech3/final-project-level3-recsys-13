FROM node:alpine

EXPOSE 30001
WORKDIR "/"

COPY package.json ./

RUN npm install

COPY ./ ./

CMD ["npm", "start"]