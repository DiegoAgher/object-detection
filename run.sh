
source /opt/conda/bin/activate obj_rec

python -m preprocessing.parse_data.to_csv
jupyter notebook --ip 0.0.0.0 --allow-root --no-browser

