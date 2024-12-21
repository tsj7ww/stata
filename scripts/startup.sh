git clone https://github.com/tsj7ww/halsey.git
cd halsey

docker-compose down
docker-compose build --no-cache # jupyter-lab
docker-compose up # jupyter-lab