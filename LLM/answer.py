import os

from google import  genai
from dotenv import load_dotenv
from google.genai.types import GenerateContentConfig, FunctionCallingConfig, ToolConfig, FunctionCallingConfigMode
from LLM.utils.tools import tools


load_dotenv()




class Response:

    def __init__(self):
        """Initialize the Gemini client and set up the configuration."""
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.configs = self.configuration(all_tools= tools)


    @staticmethod
    def configuration(all_tools : list):
        """Class that will deal with the Gemini Parameters"""


        system_prompt = ""

        config = GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
            max_output_tokens=256,
            tools=all_tools,
            tool_config=ToolConfig(function_calling_config=FunctionCallingConfig(mode=FunctionCallingConfigMode.AUTO)),

        )
        return config

    def answer(self, text : str) -> str:
        response = self.client.models.generate_content(
            model="gemini-2.0-flash", contents=text,
            config=self.configs,

        )
        return response.text


print(Response().answer("Hello, how are you?"))





