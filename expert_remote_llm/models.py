from typing import Literal

from pydantic import BaseModel


ChatRole = Literal["system", "user", "assistant"]


class ChatBlock(BaseModel):
    role: ChatRole
    content: str
    image_b64: str | None = None

    def dump_for_prompt(self) -> dict:
        if not self.image_b64:
            return self.model_dump()
        # o/w we have to change the format a bit
        return {
            "role": self.role,
            "content": [
                {"type": "text", "text": self.content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.image_b64}",
                    },
                },
            ],
        }

    pass
