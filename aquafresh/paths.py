import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS=os.environ.get('AQUAFRESH_MODELS',os.path.join(ROOT,'models'))
RESULTS=os.path.join(ROOT,'results'); FIGURES=os.path.join(ROOT,'figures')
for _d in (MODELS,RESULTS,FIGURES): os.makedirs(_d,exist_ok=True)
