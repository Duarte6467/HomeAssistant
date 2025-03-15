import os

from google import  genai
from dotenv import load_dotenv ;
from google.genai.types import GenerateContentConfig

load_dotenv()







class Response():

    def __init__(self):


        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


    def tools(self, tools) -> list:
        """Class that will fetch and load the tools that are going to be used in the Gemini API"""




        return list(tools)


    def configuration(self):
        """Class that will deal with the Gemini Parameters"""


        system_prompt = ""

        config = GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
            max_output_tokens=256,
            tools=[]
        )

    def answer(self, text : str) -> str:
        response = self.client.models.generate_content(
            model="gemini-2.0-flash", contents=text,
        )

        return response.text



