from google.genai import types
from google import genai


tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="changeLights",
                description="Change light brightness, colour of specific rooms based on user input ",
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    required=["bulb"],
                    properties={
                        "brightness": genai.types.Schema(
                            type=genai.types.Type.NUMBER,
                        ),
                        "colour": genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                        "bulb": genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                        "division": genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    },
                ),
            ),
        ])
]