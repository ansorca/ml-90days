from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    data: list[list[float]]
     
    @field_validator('data')
    @classmethod
    def check_feature_count(cls, v):
        for sample in v:
            if len(sample) != 4:
                raise ValueError(f"Expected 4 features, got {len(sample)}")
        return v