docker-credential-gcr configure-docker
git pull
cd backend
docker build . -t gcr.io/stoked-magpie-351406/bedep
cd ../frontend
docker build . -t gcr.io/stoked-magpie-351406/fedep
docker push gcr.io/stoked-magpie-351406/bedep
docker push gcr.io/stoked-magpie-351406/fedep