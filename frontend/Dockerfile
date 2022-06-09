FROM node:lts-alpine3.16

EXPOSE 30001
WORKDIR "/"

COPY package.json ./

RUN npm install 
RUN npm install --force --save react-swipeable-views

COPY ./ ./

CMD ["npm", "start"]