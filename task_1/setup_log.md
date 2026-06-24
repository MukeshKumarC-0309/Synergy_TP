# Setup Log — Task 1

## Folder creation
mkdir -p task_1/src task_1/data

## Virtual environment
cd task_1
python3 -m venv venv
source venv/bin/activate

## Install packages
pip install requests

## Generate requirements.txt
pip freeze > requirements.txt

## Git commands
cd ..
git add .
git commit -m "Add Task 1 setup"
git push origin main