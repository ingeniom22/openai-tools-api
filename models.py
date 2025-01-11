from pydantic import BaseModel, Field


class KGRetrieverInput(BaseModel):
    question: str = Field(description="User questions")