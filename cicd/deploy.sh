docker pull gcr.io/stoked-magpie-351406/fedep
docker pull gcr.io/stoked-magpie-351406/bedep
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$PWD:$PWD" -w="$PWD" docker/compose:1.27.4 stop
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$PWD:$PWD" -w="$PWD" docker/compose:1.27.4 up -d