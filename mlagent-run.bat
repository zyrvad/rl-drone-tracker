cd "C:\Projects\Terrain Generation"
call venv\Scripts\activate

mlagents-learn --run-id=test_obstacles --results-dir=test_obstacles config_obstacles.yaml --force