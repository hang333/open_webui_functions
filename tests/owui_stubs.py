"""
Lightweight stand-ins for the Open WebUI modules that the Gemini Manifold plugins import.

The signatures mirror Open WebUI v0.13.4 (backend/open_webui/models/chats.py, models/files.py,
models/functions.py, models/users.py, storage/provider.py, utils/misc.py and
utils/access_control/files.py): methods that are async there are async here. The tables keep their data in
memory, so a test can arrange a chat, its files and the installed functions and inspect what the plugin wrote.

`install()` registers the modules in `sys.modules` before the plugins are imported (see conftest.py);
`fresh_modules()` returns new instances for a test to patch into a plugin module.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, BinaryIO

from pydantic import BaseModel


@dataclass
class ChatModel:
    id: str
    user_id: str
    chat: dict[str, Any]


@dataclass
class FileModel:
    id: str
    user_id: str
    path: str
    meta: dict[str, Any] | None = None
    data: dict[str, Any] = field(default_factory=dict)
    filename: str = ''


@dataclass
class FunctionModel:
    id: str
    is_active: bool = True
    is_global: bool = False


@dataclass
class UserModel:
    id: str
    role: str = 'user'
    email: str = ''


class FileForm(BaseModel):
    id: str
    hash: str | None = None
    filename: str
    path: str
    data: dict = {}
    meta: dict = {}


class ChatsTable:
    def __init__(self) -> None:
        self.chats: dict[str, ChatModel] = {}
        self.chat_files: list[dict[str, Any]] = []
        self.upserts: list[tuple[str, str, dict]] = []

    def add_chat(self, chat_id: str, user_id: str, messages: dict[str, dict], current_id: str) -> None:
        self.chats[chat_id] = ChatModel(chat_id, user_id, {'history': {'messages': messages, 'currentId': current_id}})

    async def get_chat_by_id(self, id: str, db: Any = None) -> ChatModel | None:
        return self.chats.get(id)

    async def get_chat_by_id_and_user_id(self, id: str, user_id: str, db: Any = None) -> ChatModel | None:
        chat = self.chats.get(id)
        return chat if chat and chat.user_id == user_id else None

    async def get_messages_map_by_chat_id(self, id: str) -> dict | None:
        chat = self.chats.get(id)
        return chat.chat['history']['messages'] if chat else None

    async def is_chat_owner(self, id: str, user_id: str, db: Any = None) -> bool:
        chat = self.chats.get(id)
        return bool(chat and chat.user_id == user_id)

    async def get_accessible_chat_by_id(
        self, id: str, user: Any, db: Any = None, *, permission: str = 'read', chat: Any = None
    ) -> ChatModel | None:
        chat = chat or self.chats.get(id)
        return chat if chat and chat.user_id == user.id else None

    async def upsert_message_to_chat_by_id_and_message_id(
        self, id: str, message_id: str, message: dict, *, touch: bool = True
    ) -> ChatModel | None:
        self.upserts.append((id, message_id, message))
        return self.chats.get(id)

    async def add_message_files_by_id_and_message_id(self, id: str, message_id: str, files: list[dict]) -> list[dict]:
        return list(files)

    async def get_message_metadata(self, chat_id: str, message_id: str, metadata_key: str) -> Any | None:
        return None

    async def insert_chat_files(
        self, chat_id: str, message_id: str, file_ids: list[str], user_id: str, db: Any = None
    ) -> list | None:
        rows = [{'chat_id': chat_id, 'message_id': message_id, 'file_id': f, 'user_id': user_id} for f in file_ids]
        self.chat_files.extend(rows)
        return rows


class FilesTable:
    def __init__(self) -> None:
        self.files: dict[str, FileModel] = {}

    async def get_file_by_id(self, id: str, db: Any = None) -> FileModel | None:
        return self.files.get(id)

    async def insert_new_file(self, user_id: str, form_data: FileForm, db: Any = None) -> FileModel | None:
        model = FileModel(form_data.id, user_id, form_data.path, form_data.meta, form_data.data, form_data.filename)
        self.files[model.id] = model
        return model


class FunctionsTable:
    def __init__(self) -> None:
        self.functions: dict[str, FunctionModel] = {}

    async def get_function_by_id(self, id: str, db: Any = None) -> FunctionModel | None:
        return self.functions.get(id)


class UsersTable:
    def __init__(self) -> None:
        self.users: dict[str, UserModel] = {}

    async def get_user_by_id(self, id: str, db: Any = None) -> UserModel | None:
        return self.users.get(id)


class StorageProvider:
    """Local storage provider: `upload_file` keeps the bytes, `get_file` returns a local path."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.local_paths: dict[str, str] = {}

    def upload_file(self, file: BinaryIO, filename: str, tags: dict[str, str]) -> tuple[bytes, str]:
        contents = file.read()
        path = f'/app/backend/data/uploads/{filename}'
        self.blobs[path] = contents
        return contents, path

    def get_file(self, file_path: str) -> str:
        return self.local_paths.get(file_path, file_path)


class AccessControl:
    """`has_access_to_file` grants read access to the (file id, user id) pairs in `grants`."""

    def __init__(self) -> None:
        self.grants: set[tuple[str, str]] = set()

    async def has_access_to_file(
        self,
        file_id: str | None,
        access_type: str,
        user: Any,
        db: Any = None,
        user_group_ids: set[str] | None = None,
    ) -> bool:
        return (file_id, user.id) in self.grants


def pop_system_message(messages: list[dict]) -> tuple[dict | None, list[dict]]:
    system = next((message for message in messages if message.get('role') == 'system'), None)
    return system, [message for message in messages if message.get('role') != 'system']


def get_message_list(messages_map: dict, message_id: str) -> list[dict]:
    message_list = []
    current = messages_map.get(message_id) if messages_map else None
    seen = set()
    while current and message_id not in seen:
        seen.add(message_id)
        message_list.append(current)
        message_id = current.get('parentId')
        current = messages_map.get(message_id) if message_id else None
    return list(reversed(message_list))


@dataclass
class OwuiModules:
    """One set of stub instances, as a plugin module sees them after `patch_plugin`."""

    chats: ChatsTable = field(default_factory=ChatsTable)
    files: FilesTable = field(default_factory=FilesTable)
    functions: FunctionsTable = field(default_factory=FunctionsTable)
    users: UsersTable = field(default_factory=UsersTable)
    storage: StorageProvider = field(default_factory=StorageProvider)
    access: AccessControl = field(default_factory=AccessControl)

    def patch_plugin(self, monkeypatch: Any, module: Any) -> OwuiModules:
        """Points the Open WebUI names a plugin module imported at these instances."""
        replacements = {
            'Chats': self.chats,
            'Files': self.files,
            'FileForm': FileForm,
            'Functions': self.functions,
            'Users': self.users,
            'Storage': self.storage,
            'has_access_to_file': self.access.has_access_to_file,
            'pop_system_message': pop_system_message,
        }
        for name, value in replacements.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, value)
        return self


def install() -> None:
    """Registers the stub modules (only where no other test registered its own)."""
    default = OwuiModules()
    modules = {
        'open_webui': {},
        'open_webui.models': {},
        'open_webui.models.chats': {'Chats': default.chats},
        'open_webui.models.files': {'Files': default.files, 'FileForm': FileForm},
        'open_webui.models.functions': {'Functions': default.functions},
        'open_webui.models.users': {'Users': default.users},
        'open_webui.storage': {},
        'open_webui.storage.provider': {'Storage': default.storage},
        'open_webui.utils': {},
        'open_webui.utils.misc': {'pop_system_message': pop_system_message, 'get_message_list': get_message_list},
        'open_webui.utils.access_control': {},
        'open_webui.utils.access_control.files': {'has_access_to_file': default.access.has_access_to_file},
    }
    for name, attributes in modules.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        if any(other.startswith(f'{name}.') for other in modules):
            module.__path__ = []  # A package, so that its submodules can be imported.
        sys.modules[name] = module
