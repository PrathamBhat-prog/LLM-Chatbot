from dotenv import load_dotenv
import os

# Load environment variables from .env (only once in the app)
load_dotenv()


class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    # default model is gemini-2.5-flash, but you can override in .env
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

    def validate(self) -> None:
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in .env")


# a single settings instance to import everywhere
settings = Settings()
