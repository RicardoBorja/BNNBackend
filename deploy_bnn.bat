@echo off
REM Restore all files to the latest commit
git restore .
git restore --staged .
REM Pull latest changes from remote
git pull
REM Build Docker image with name bnn
docker build -t bnn .
REM Stop and remove any existing container named bnn
docker stop bnn 2>NUL
REM Remove the container if it exists
docker rm bnn 2>NUL
REM Run the Docker image, mapping port 8081
docker run -d --name bnn -p 8081:8080 bnn
