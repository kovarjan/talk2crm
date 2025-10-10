import datetime
import textwrap
import json
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from core.config import CRM_SYSTEM, CRM_INSTANCE

class ChatSession:
    def __init__(self, init: bool = True, system_prompt: str = None):
        self.messages = []
        if init:
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_day = datetime.datetime.now().strftime("%A")
            
            self.add_system(
                f"You are a {CRM_SYSTEM} CRM assistant. Your task is to generate JSON commands for CRM according to the user's voice instructions. Act as a helpful female assistant who provides clear and concise answers. "
                "Answer to user via message_to_user in Czech language. Provide clarifications if needed. Use the given schema. Use correct czech declension of name. Be polite.\n\n "
                f"CRM system: {CRM_SYSTEM}\n"
                f"Client instance: {CRM_INSTANCE}\n"
                f"{system_prompt} \n"
                f"Current date and time: {current_datetime} and today is {current_day}\n "
                "If any key information is missing, respond with a question to clarify.\n "
                "ALWAYS respond in a single valid JSON format with correct schema! No json comments, include your points in property message_to_user\n"
                "Absolutely no text or commentary between the last Observation and \"Final Answer:\". If you include any other text, the run fails.\n"
            )
        elif system_prompt:
            self.add_system(system_prompt)

    def clone(self):
        """
        Creates a clone of the current chat session.
        """
        new_session = ChatSession(init=False)
        new_session.messages = self.messages.copy()
        return new_session

    def reset(self):
        """
        Resets the chat session by clearing all messages.
        """
        self.messages = []
        print("Chat session has been reset.")

    def load_history(self, history: list):
        """
        Imports a list of messages into the chat session.
        Each message should be a dictionary with 'role' and 'content'.
        """
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                self.messages.append(msg)
            else:
                raise ValueError("Each message must be a dictionary with 'role' and 'content'.")

    def add_user(self, content: str):
        # check if ist's composed message (separated by pipe) and if so, split it and add only the last part
        if "| " in content:
            content = content.split("| ")[-1].strip()
        self.messages.append({"role": "user", "content": content})

    def add_system(self, content: str):
        self.messages.append({"role": "system", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def inject_context(self, label: str, name: str, id: str):
        self.add_assistant(f"{label} from CRM:\n- Name: {name}\n- ID: {id}")

    def add_llm_response(self, response: dict, format="ollama"):
        """
        Adds the LLM response to the assistant role with proper formatting.
        `format` options:
            - 'ollama': JSON escaped string
            - 'raw': Pretty JSON block (OpenAI-style)
        """
        if format == "ollama":
            pretty = json.dumps(response, indent=4, ensure_ascii=False)
            self.add_assistant(f"```json\n{pretty}\n```")
        elif format == "raw":
            json_string = json.dumps(response, ensure_ascii=False)
            self.add_assistant(json_string)
        else:
            raise ValueError("Unsupported format: choose 'ollama' or 'raw'")

    def prepend_system(self, content: str):
        """
        Prepend a system message to the chat session.
        """
        self.messages.insert(0, {"role": "system", "content": content})

    def compose_following_user_message(self, user_message: str):
        """
        Compose following user message, join all previous user messages into one string separated by comma.
        """
        user_messages = self.get_user_messages()
        if not user_messages:
            return user_message
        
        # Join all previous user messages into one string
        previous_user_messages = " | ".join(msg["content"] for msg in user_messages)
        return f"{previous_user_messages} | {user_message}" if previous_user_messages else user_message

    def get_messages(self):
        return self.messages
    
    def get_count(self):
        """
        Returns the number of messages in the chat session.
        """
        return len(self.messages)
    
    def get_last_message(self):
        """
        Returns the last message in the chat session.
        If the session is empty, returns None.
        """
        if self.messages:
            return self.messages[-1]
        return None
    
    def get_user_messages(self):
        """
        Returns a list of all user messages in the chat session.
        """
        return [msg for msg in self.messages if msg["role"] == "user"]
    
    def get_user_assistant_messages(self):
        """
        Returns a list of all messages from the user and the assistant in the chat session.
        """
        output = []
        for msg in self.messages:
            if msg["role"] == "user":
                output.append(msg)
            elif msg["role"] == "assistant":
                try:
                    content_json = json.loads(msg["content"]) if isinstance(msg["content"], str) else msg["content"]
                    if isinstance(content_json, dict) and "message_to_user" in content_json:
                        # Copy message and replace content with message_to_user
                        output.append({
                            "role": "assistant",
                            "content": content_json["message_to_user"]
                        })
                except Exception:
                    continue

        return output
    
    def get_module_data(self):
        """
        Find message from ModuleDataExtractor with this text and return module and action from it.
        example:
        ModuleDataExtractor - user request context: 
            - Module: 'meetings'
            - Action: 'create'
        """
        for msg in self.messages:
            if msg["role"] == "assistant":
                content = msg["content"]
                print(f" --> get_module_data --> Checking message: {content}")
                if "ModuleDataExtractor - user request context:" in content:
                    lines = content.split("\n")
                    module = None
                    action = None
                    for line in lines:
                        if "Module:" in line:
                            module = line.split("Module:")[1].strip().strip("'")
                        elif "Action:" in line:
                            action = line.split("Action:")[1].strip().strip("'")
                    if module and action:
                        return {"module": module, "action": action}
        return {}
    
    def to_langchain_messages(self):
        langchain_messages = []
        for msg in self.get_messages():
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))

        return langchain_messages

    def pretty_print(self, return_as_string=False, indent=2):
        COLORS = {
            "USER": "\033[94m",  # Blue
            "ASSISTANT": "\033[92m",  # Green
            "SYSTEM": "\033[90m",  # Grey
            "RESET": "\033[0m",
        }

        out_lines = ["\n🔁 Chat History:\n"]

        indent_str = " " * indent

        for msg in self.messages:
            role = msg["role"].upper()
            color = COLORS.get(role, "")
            reset = COLORS["RESET"]
            header = f"{indent_str}{color}--- {role} ---{reset}"
            wrapped_content = textwrap.indent(
                textwrap.fill(str(msg["content"]), width=120), prefix=indent_str
            )
            out_lines.append(f"{header}\n{wrapped_content}")

        output = "\n".join(out_lines)
        if return_as_string:
            return output
        else:
            print(output)
