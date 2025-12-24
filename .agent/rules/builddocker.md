---
trigger: always_on
---

After a change is commited and pushed to the "mobile-redesign" branch, build the docker container as follows:

docker buildx build --platform linux/amd64 -t ghcr.io/akeslo/podcast-ad-remover:mobile-redesign --push .

Once uploaded, remove any prior builds from ghcr.io.