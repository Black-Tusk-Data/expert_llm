from typing import Literal

from pydantic import BaseModel


ChatRole = Literal["system", "user", "assistant"]


class ChatBlock(BaseModel):
    role: ChatRole
    content: str
    pass
