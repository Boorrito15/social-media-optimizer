

### run at comp

```python
# First-time full fit (~15min)
python -m models.train --task all --retrain-state --retrain
```





### use

```python
import json
import pandas as pd
from models import predict_lin, predict_clas

dummy_row = pd.DataFrame([{
    # all dumies unless stated otherwise
    "campaign": "Organic | Website",
    "year": 2025,
    "page": "ABXV",                                         # not dummy
    "platform": "FB",                                       # not dummy
    "media_type": "Short Video",
    "category_l0": "No Hashtag",
    "category_l1": "No Hashtag",
    "category_l2": "No Hashtag",
    "url": "https://example.com/dummy-reel",
    "content": "@jacobkneepkens crosses the chalk 🤙",      # not dummy
    "cost_nzd": None,
    "views": 123,                         
    "engagement": 456,                     
    "hours": 64.0,
    # not dummy
    "description_json": json.dumps({
        "play_by_play": "A male rugby player sprints downfield, breaks a tackle and scores a try.",
        "content_theme": ["celebration"],
        "format_access": ["highlight"],
        "people": ["dummy player"],
        "brands": [],
        "event": [],
        "tone": ["excitement"],
        "context": ["stadium"],
        "overall_team": ["men"],
        "audio_format": ["ambient"],
    }),
    "duration_seconds": 12.0, # not dummy
}])


lin_out  = predict_lin(dummy_row)    # [{"engagement": <float>, "views": <float>}]
clas_out = predict_clas(dummy_row)   # [{"engagement": "engagement_<i>", "views": "views_<j>"}]

print(lin_out)
print(clas_out)
```