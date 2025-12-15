"""Local Planner (Activity to Action) LLM class for handling interactions with language models."""
import base64
import io
import json
import re
import time

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel

from simworld.utils.logger import Logger

from .base_llm import BaseLLM


class A2ALLM(BaseLLM):
    """Local Planner (Activity to Action) LLM class for handling interactions with language models."""
    def __init__(self, model_name: str = 'gpt-5-nano', url: str = None, provider: str = 'openai'):
        """Initialize the Local Planner LLM."""
        super().__init__(model_name, url, provider)

        self.logger = Logger.get_logger('A2ALLM')

    def generate_instructions(self, system_prompt, user_prompt, images=[], max_tokens=None, temperature=0.7, top_p=1.0, response_format=BaseModel):
        """Generate instructions for the Local Planner system.

        Args:
            system_prompt (str): The system prompt for the Local Planner system.
            user_prompt (str): The user prompt for the Local Planner system.
            images (list): The images for the Local Planner system.
            max_tokens (int): The maximum number of tokens for the Local Planner system.
            temperature (float): The temperature for the Local Planner system.
            top_p (float): The top_p for the Local Planner system.
            response_format (BaseModel): The response format for the Local Planner system.
        """
        if self.provider == 'openai':
            return self._generate_instructions_openai(system_prompt, user_prompt, images, max_tokens, temperature, top_p, response_format)
        elif self.provider == 'openrouter':
            return self._generate_instructions_openrouter(system_prompt, user_prompt, images, max_tokens, temperature, top_p, response_format)
        else:
            raise ValueError(f'Invalid provider: {self.provider}')

    def _generate_instructions_openai(self, system_prompt, user_prompt, images=[], max_tokens=None, temperature=0.7, top_p=1.0, response_format=BaseModel):
        start_time = time.time()
        
        if images:
            self.logger.warning("Images not supported in GPT-5 Responses API migration, ignoring images")
        
        user_content = user_prompt

        try:
            reasoning_effort = "minimal"
            text_verbosity = "low"
            
            text_format = None
            if response_format and response_format != BaseModel:
                if hasattr(response_format, '__name__'):
                    if response_format.__name__ == 'HighLevelActionSpace':
                        simplified_schema = {
                            "type": "object",
                            "properties": {
                                "action_queue": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "A list of action indices to be performed"
                                },
                                "destination": {
                                    "anyOf": [
                                        {
                                            "type": "array", 
                                            "items": {"type": "number"},
                                            "minItems": 2,
                                            "maxItems": 2
                                        },
                                        {"type": "null"}
                                    ],
                                    "description": "The [x, y] coordinates of the destination, or null if not needed"
                                },
                                "object_name": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "null"}
                                    ],
                                    "description": "The name of the object to interact with, or null if not needed"
                                },
                                "reasoning": {
                                    "type": "string", 
                                    "description": "The reasoning for the chosen actions"
                                }
                            },
                            "required": ["action_queue", "destination", "object_name", "reasoning"],
                            "additionalProperties": False
                        }
                    elif response_format.__name__ == 'LowLevelActionSpace':
                        simplified_schema = {
                            "type": "object",
                            "properties": {
                                "choice": {
                                    "type": "integer",
                                    "enum": [0, 1, 2],
                                    "description": "The choice of action: 0=do nothing, 1=step, 2=turn"
                                },
                                "duration": {
                                    "anyOf": [
                                        {"type": "number"},
                                        {"type": "null"}
                                    ],
                                    "description": "Duration in seconds for step action, or null if not applicable"
                                },
                                "direction": {
                                    "anyOf": [
                                        {"type": "integer"},
                                        {"type": "null"}
                                    ],
                                    "description": "Direction for step action: 0=forward, 1=backward, or null if not applicable"
                                },
                                "angle": {
                                    "anyOf": [
                                        {"type": "number"},
                                        {"type": "null"}
                                    ],
                                    "description": "Angle in degrees for turn action, or null if not applicable"
                                },
                                "clockwise": {
                                    "anyOf": [
                                        {"type": "boolean"},
                                        {"type": "null"}
                                    ],
                                    "description": "Turn direction: true=clockwise, false=counterclockwise, or null if not applicable"
                                },
                                "reasoning": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "null"}
                                    ],
                                    "description": "The reasoning for the chosen action, or null if not provided"
                                }
                            },
                            "required": ["choice", "duration", "direction", "angle", "clockwise", "reasoning"],
                            "additionalProperties": False
                        }
                    else:
                        original_schema = response_format.model_json_schema()
                        simplified_schema = original_schema.copy()
                        if "additionalProperties" not in simplified_schema:
                            simplified_schema["additionalProperties"] = False
                else:
                    simplified_schema = response_format.model_json_schema()
                    if "additionalProperties" not in simplified_schema:
                        simplified_schema["additionalProperties"] = False
                
                text_format = {
                    "type": "json_schema",
                    "name": getattr(response_format, '__name__', 'response'),
                    "strict": True,
                    "schema": simplified_schema
                }
            
            # Debug the input format
            self.logger.info(f'Input type: {type(user_content)}, Input content: {user_content}')
            self.logger.info(f'Text format: {text_format}')
            
            response = self.client.responses.create(
                model=self.model_name,
                instructions=system_prompt,
                input=user_content,
                max_output_tokens=max_tokens,
                reasoning={"effort": reasoning_effort},
                text={"verbosity": text_verbosity, "format": text_format} if text_format else {"verbosity": text_verbosity},
            )
            
            if text_format:
                action_json = response.output_text
                if isinstance(action_json, str):
                    import json
                    try:
                        action_json = json.loads(action_json)
                        self.logger.info(f'Successfully parsed JSON: {action_json}')
                    except json.JSONDecodeError as e:
                        self.logger.error(f'Failed to parse structured output as JSON: {e}')
                        self.logger.error(f'Raw content: {action_json}')
                        action_json = None
            else:
                action_json = response.output_text
        except Exception as e:
            self.logger.error(f'Error in generate_instructions_openai: {e}')
            action_json = None

        return action_json, time.time() - start_time

    def _generate_instructions_openrouter(self, system_prompt, user_prompt, images=[], max_tokens=None, temperature=0.7, top_p=1.0, response_format=BaseModel):

        start_time = time.time()
        
        if response_format and response_format != BaseModel:
            user_prompt += '\nPlease respond in valid JSON format following this schema: ' + str(response_format.model_json_schema())
        
        if images:
            user_content = [{
                'role': 'user',
                'content': []
            }]
            
            user_content[0]['content'].append({
                'type': 'text',
                'text': user_prompt
            })
            
            for image in images:
                img_data = self._process_image_to_base64(image)
                user_content[0]['content'].append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{img_data}'}
                })
        else:
            user_content = user_prompt

        self.logger.info(f'user_content: {user_content}')

        action_response = None
        try:
            reasoning_effort = "minimal"
            text_verbosity = "low"
            
            response = self.client.responses.create(
                model=self.model_name,
                instructions=system_prompt,
                input=user_content,
                max_output_tokens=max_tokens,
                reasoning={"effort": reasoning_effort},
                text={"verbosity": text_verbosity},
            )
            action_response = response.output_text
        except Exception as e:
            self.logger.error(f'Error in generate_instructions_openrouter: {e}')
            action_response = None

        if action_response is None:
            self.logger.warning('Warning: Failed to get action response, using default')
            action_json = None
        else:
            action_json = self._extract_json_and_fix_escapes(action_response)

        return action_json, time.time() - start_time

    def _process_image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string.

        Args:
            image (np.ndarray): Image array (1 or 3 channels)

        Returns:
            str: Base64 encoded image string
        """
        # Convert single channel to 3 channels if needed
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Ensure uint8 type
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Convert to PIL Imagemedium
        pil_image = Image.fromarray(image)

        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format='JPEG')
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str

    def _extract_json_and_fix_escapes(self, text):
        # Extract content from first { to last }
        pattern = r'(\{.*\})'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            json_str = match.group(1)
            # Fix invalid escape sequences in JSON
            fixed_json = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
            try:
                # Try to parse the fixed JSON
                json_obj = json.loads(fixed_json)
                return json_obj
            except json.JSONDecodeError as e:
                self.logger.error(f'JSON parsing error: {e}')
                self.logger.error(f'Fixed JSON: {fixed_json}')
                # Return the fixed string if parsing fails
                return fixed_json
        else:
            self.logger.error(f'No JSON found in the text: {text}')
            return None
