# sdcguard in a container (build from the repo root: docker build -f deploy/Dockerfile -t sdcguard .)
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime
COPY pyproject.toml README.md LICENSE /opt/sdcguard/
COPY src /opt/sdcguard/src
RUN pip install --no-cache-dir "/opt/sdcguard[certify,metrics]"
ENTRYPOINT ["sdcguard"]
CMD ["nhc", "--minutes", "2"]
