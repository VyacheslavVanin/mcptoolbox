import asyncio
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any

from core.configuration import Configuration
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .tool import Tool


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            read, write = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


# contains Servers
class ToolBox:
    def __init__(self, server_config: dict[str, Any]):
        self.servers = [
            Server(name, srv_config)
            for name, srv_config in server_config["mcpServers"].items()
        ]
        self.tools: dict[str, list[Any]] = dict()
        self.tools_description = ""
        self.inited = False

    async def initialize(self) -> bool:
        """Initialize all servers in the toolbox.

        Returns:
            bool: True if all servers initialized successfully, False otherwise
        """
        if self.inited:
            return True

        try:
            for server in self.servers:
                await server.initialize()
                self.tools[server.name] = await server.list_tools()
        except Exception as e:
            logging.error(f"Failed to initialize server: {e}")
            await self.cleanup_servers()
            return False
        self.inited = True
        return True

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]):
        for server in self.servers:
            tools = self.tools[server.name]
            if any(tool.name == tool_name for tool in tools):
                return await server.execute_tool(tool_name, tool_args)
        return None

    def get_tools_descriptions(self) -> str:
        if not self.tools_description:
            all_tools = []
            for server in self.servers:
                tools = self.tools[server.name]
                all_tools.extend(tools)
            self.tools_description = "\n".join(
                [tool.format_for_llm() for tool in all_tools]
            )
        return self.tools_description

    # returns tools in openai format like this:
    # [
    #   {
    #     "type": "function",
    #     "function": {
    #       "name": "get_current_weather",
    #       "description": "Get the current weather in a given location",
    #       "parameters": {
    #         "type": "object",
    #         "properties": {
    #           "location": {
    #             "type": "string",
    #             "description": "The city and state, e.g. San Francisco, CA"
    #           },
    #           "unit": {
    #             "type": "string",
    #             "enum": ["celsius", "fahrenheit"]
    #           }
    #         },
    #         "required": ["location"]
    #       }
    #     }
    #   }
    # ]
    def get_tools(self) -> list[dict[str, Any]]:
        tools_list: list[dict[str, Any]] = []
        for server_name, tools in self.tools.items():
            for tool in tools:
                tools_list.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return tools_list
