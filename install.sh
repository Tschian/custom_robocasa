pip install -e custom_robosuite/
pip install -e custom_robocasa/

pip install -r requirements.txt

python -m robocasa.scripts.download_kitchen_assets

python -m robosuite.scripts.setup_macros
python -m robocasa.scripts.setup_macros