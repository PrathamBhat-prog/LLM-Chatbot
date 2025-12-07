import google.generativeai as genai

from src.config.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def main() -> None:
    settings.validate()
    logger.info("Starting Gemini test call...")
    logger.info(f"Using model: {settings.GEMINI_MODEL}")

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(settings.GEMINI_MODEL)

    prompt = "Hello Gemini! Just respond with a short motivational quote."
    logger.info("Sending prompt to Gemini API")

    response = model.generate_content(prompt)

    logger.info("Received response from Gemini")
    print("\nGemini Response:\n")
    print(response.text)


if __name__ == "__main__":
    main()
