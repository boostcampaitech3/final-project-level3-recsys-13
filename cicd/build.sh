docker-credential-gcr configure-docker
cd ../backend
docker build . -t gcr.io/recipe-reco/bedep
cd ../frontend
docker build . -t gcr.io/recipe-reco/fedep
docker push gcr.io/recipe-reco/bedep
docker push gcr.io/recipe-reco/fedep