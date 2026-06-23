import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ai.model_versioning import ModelVersioning

versions_dir = 'models/versions'
highest_v = 0
for f in os.listdir(versions_dir):
    match = re.match(r'model_v(\d+)\.pt', f)
    if match:
        v = int(match.group(1))
        if v > highest_v:
            highest_v = v

if highest_v > 0:
    mv = ModelVersioning()
    mv.promote_to_live(highest_v)
    print(f'Promoted v{highest_v} to live')
else:
    print('No versions found to promote.')
