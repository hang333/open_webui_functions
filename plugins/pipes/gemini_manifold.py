"""
title: Gemini Manifold Plus
id: gemini_manifold_plus
description: Gemini Manifold google_genai 2.1.0, Plus edition: native tools, Gemini 4 thinking, Interactions API.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 2.2.0+plus.4
requirements: google-genai==3.3.0
"""

from __future__ import annotations

import asyncio
import base64
import copy
import datetime as dt
import difflib
import fnmatch
import hashlib
import importlib.metadata
import io
import json
import mimetypes
import re
import sys
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from functools import cache, cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    cast,
    get_args,
    get_origin,
)
from urllib.parse import parse_qs, urlparse

import aiofiles
import pydantic_core
import xxhash
import yaml
from aiocache import cached
from aiocache.backends.memory import SimpleMemoryCache
from aiocache.base import BaseCache
from aiocache.serializers import NullSerializer
from fastapi import FastAPI, Request
from fastapi.datastructures import State
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    create_model,
    field_validator,
)

# I change these only when I make a release to avoid PR merge conflicts.
# If you are making a PR then please do not change these values.
VERSION = '2.2.0+plus.4'
# This is the recommended version for the companion filter.
# Older versions might still work, but backward compatibility is not guaranteed
# during the development of this personal use plugin.
RECOMMENDED_COMPANION_VERSION = '2.2.0'


# Keys `title`, `id` and `description` in the frontmatter above are used for my own development purposes.
# They don't have any effect on the plugin's functionality.
# The Plus edition has its own function id, so it can be installed next to "Gemini Manifold google_genai".


# This is a helper function that provides a manifold for Google's Gemini Studio API and Vertex AI.
# Be sure to check out my GitHub repository for more information!
# Contributions, questions and suggestions are very welcome.


def _unload_stale_modules() -> None:
    """
    Open WebUI pip installs frontmatter `requirements` into the already running server, which can replace
    packages on disk while their old versions are still in `sys.modules`. Importing `google.genai` would then
    mix modules from both versions and fail, e.g. with `cannot import name 'OP_BINARY' from 'websockets.frames'`
    (`google-genai` requires `websockets<18`, so pip may downgrade it) or with
    `module 'google.genai.types' has no attribute ...` (`google-genai` itself got upgraded).
    Dropping the stale modules makes the import load a consistent set of modules from disk.
    Code that already imported the old modules (e.g. uvicorn) keeps its own references to them.
    """
    unloaded = False
    # Distribution name, package name and the module whose `__version__` tells which version is loaded.
    for distribution, package, version_module in (
        ('websockets', 'websockets', 'websockets'),
        ('google-genai', 'google.genai', 'google.genai.version'),
    ):
        loaded = sys.modules.get(version_module)
        if loaded is None:
            continue
        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        if getattr(loaded, '__version__', None) == installed_version:
            continue
        for name in [n for n in sys.modules if n == package or n.startswith(f'{package}.')]:
            del sys.modules[name]
        # `from google import genai` would otherwise still return the old module set on the parent package.
        parent_name, _, child = package.rpartition('.')
        parent = sys.modules.get(parent_name)
        if parent is not None and hasattr(parent, child):
            delattr(parent, child)
        unloaded = True
    if unloaded:
        importlib.invalidate_caches()


# Must run before the `google.*` and `open_webui.*` imports below: those modules
# (or their transitive dependencies) may be stale in `sys.modules` after Open
# WebUI pip-installs this plugin's frontmatter `requirements` into the running
# server. See `_unload_stale_modules` above.
_unload_stale_modules()

# These imports are intentionally below `_unload_stale_modules()` (E402 exempted
# inline): the `google.*` modules must only be imported after the stale-module
# purge, and the `open_webui.*` imports can transitively import `google.*`
# themselves (e.g. the storage provider imports `google.cloud.storage`).
from google import genai  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402
from google.genai import types  # noqa: E402

from open_webui.models.chats import Chats  # noqa: E402
from open_webui.models.files import FileForm, Files  # noqa: E402
from open_webui.models.functions import Functions  # noqa: E402
from open_webui.models.users import Users  # noqa: E402
from open_webui.storage.provider import Storage  # noqa: E402
from open_webui.utils.access_control.files import has_access_to_file  # noqa: E402
from open_webui.utils.misc import pop_system_message  # noqa: E402

# This block is skipped at runtime.
if TYPE_CHECKING:
    from plugins.filters.gemini_manifold_companion import EventEmitter

    # Imports custom type definitions (TypedDicts) for static analysis purposes (mypy/pylance).
    from utils.manifold_types import (
        AssistantMessage,
        Body,
        ChatMessageTD,
        ChatObjectDataTD,
        Content,
        Features,
        FileAttachmentTD,
        ImageContent,
        Message,
        Metadata,
        ModelData,
        TextContent,
        UserData,
        UserMessage,
    )

# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main log.
log = logger.bind(auditable=False)


def _log_and_toast(
    event_emitter: EventEmitter,
    msg: str,
    level: Literal['info', 'warning', 'error'] = 'warning',
) -> None:
    """Logs `msg` at the caller's stack location and mirrors it to the front-end as a toast."""
    log.opt(depth=1).log(level.upper(), msg)
    event_emitter.emit_toast(msg, level)


# A mapping of finish reason names (str) to human-readable descriptions.
# This allows handling of reasons that may not be defined in the current SDK version.
FINISH_REASON_DESCRIPTIONS: Final[dict[str, str]] = {
    'FINISH_REASON_UNSPECIFIED': 'The reason for finishing is not specified.',
    'STOP': 'Natural stopping point or stop sequence reached.',
    'MAX_TOKENS': 'The maximum number of tokens was reached.',
    'SAFETY': 'The response was blocked due to safety concerns.',
    'RECITATION': 'The response was blocked due to potential recitation of copyrighted material.',
    'LANGUAGE': 'The response was stopped because of an unsupported language.',
    'OTHER': 'The response was stopped for an unspecified reason.',
    'BLOCKLIST': 'The response was blocked due to a word on a blocklist.',
    'PROHIBITED_CONTENT': 'The response was blocked for containing prohibited content.',
    'SPII': 'The response was blocked for containing sensitive personally identifiable information.',
    'MALFORMED_FUNCTION_CALL': 'The model generated an invalid function call.',
    'IMAGE_SAFETY': 'Generated image was blocked due to safety concerns.',
    'UNEXPECTED_TOOL_CALL': 'The model generated an invalid tool call.',
    'IMAGE_PROHIBITED_CONTENT': 'Generated image was blocked for containing prohibited content.',
    'NO_IMAGE': 'The model was expected to generate an image, but it did not.',
    'IMAGE_OTHER': (
        'Image generation stopped for other reasons, possibly related to safety or quality. '
        'Try a different image or prompt.'
    ),
    'IMAGE_RECITATION': 'Generated image was blocked due to potential recitation.',
    'TOO_MANY_TOOL_CALLS': 'The model made too many tool calls in one step.',
    'MISSING_THOUGHT_SIGNATURE': 'A function call in the request history lacked its thought signature.',
    'MALFORMED_RESPONSE': 'The model returned a malformed response.',
}

# Finish reasons that are considered normal and do not require user notification.
NORMAL_REASONS: Final[set[types.FinishReason]] = {
    types.FinishReason.STOP,
    types.FinishReason.MAX_TOKENS,
}

# Finish reasons of a round in which the model's tool call itself failed.
FAILED_TOOL_CALL_REASONS: Final[frozenset[str]] = frozenset(
    {'MALFORMED_FUNCTION_CALL', 'UNEXPECTED_TOOL_CALL', 'TOO_MANY_TOOL_CALLS', 'MISSING_THOUGHT_SIGNATURE'}
)

# These tags will be "disabled" in the response, meaning that they will not be parsed by the backend.
SPECIAL_TAGS_TO_DISABLE: Final[list[str]] = [
    'details',
    'think',
    'thinking',
    'reason',
    'reasoning',
    'thought',
    'Thought',
    '|begin_of_thought|',
    'code_interpreter',
    '|begin_of_solution|',
]
ZWS = '\u200b'


class GenaiApiError(Exception):
    """Custom exception for errors during Genai API interactions."""

    pass


class FilesAPIError(Exception):
    """Custom exception for errors during Files API operations."""

    pass


# Model families that are not chat models (speech, live, transcription, embeddings, robotics, ...) or that only
# work through the Interactions API. They are hidden from the model list; Interactions models that this pipe
# supports are offered as virtual models instead (see `InteractionsRunner`).
NON_CHAT_MODEL_PATTERNS: Final[tuple[str, ...]] = (
    '*-tts*',
    '*-live*',
    '*native-audio*',
    '*transcribe*',
    '*embedding*',
    '*robotics*',
    '*computer-use*',
    'gemini-decision-*',
    'gemini-omni-*',
    'deep-research-*',
    'antigravity-*',
    'imagen-*',
    'veo-*',
    'lyria-*',
    'genie-*',
    'aqa',
)

# Interactions API agents and models offered as virtual models (aistudio.google.com/docs/deep-research, /docs/omni).
DEEP_RESEARCH_AGENTS: Final[dict[str, str]] = {
    'deep-research-preview-04-2026': 'Deep Research (Interactions API)',
    'deep-research-max-preview-04-2026': 'Deep Research Max (Interactions API)',
}
OMNI_VIDEO_MODEL: Final[str] = 'gemini-omni-1.2-flash'

# `reasoning_details` formats. Open WebUI stores these items with the response and sends them back with the
# assistant message on later requests of the same model (aistudio.google.com/docs/thought-signatures).
# - REPLAY_FORMAT: the exact model content of one round, thought signatures included.
# - INTERACTION_FORMAT: the id of the interaction that produced the message (Interactions API models).
REPLAY_FORMAT: Final[str] = 'gemini-manifold-content-v1'
INTERACTION_FORMAT: Final[str] = 'gemini-manifold-interaction-v1'
# Documented placeholder for a function call whose thought signature is missing, e.g. a call that Open WebUI
# itself added to the history. Only function calls of the current turn are validated.
SKIP_THOUGHT_SIGNATURE: Final[str] = 'skip_thought_signature_validator'
# Text of the user message in which Open WebUI passes on the images of tool results.
OWUI_TOOL_IMAGES_TEXT: Final[str] = 'Here are the images from the tool results above. Please analyze them.'
# Gemini 3.5 Flash-Lite, 3.6 Flash and later reject a request that ends with a model turn (prefill), so a
# continued answer is requested with this user turn instead.
CONTINUE_PROMPT: Final[str] = (
    'Continue your previous answer exactly where it stopped. Do not repeat anything you already wrote.'
)
# Added when the last user turn holds only attachments: Vertex AI and Gemini 3.8+ need text in that turn.
ATTACHMENTS_ONLY_PROMPT: Final[str] = (
    'The user did not send any text message with the additional context. Answer by summarizing the newly added context.'
)


class UploadStatusManager:
    """
    Manages and centralizes status updates for concurrent file uploads.

    This manager is self-configuring. It discovers the number of files that
    require an actual upload at runtime, only showing a status message to the
    user when network activity is necessary.

    The communication protocol uses tuples sent via an asyncio.Queue:
    - ('REGISTER_UPLOAD',): Sent by a worker when it determines an upload is needed.
    - ('COMPLETE_UPLOAD',): Sent by a worker when its upload is finished.
    - ('FINALIZE',): Sent by the orchestrator when all workers are done.
    """

    def __init__(
        self,
        event_emitter: EventEmitter,
    ):
        self.event_emitter = event_emitter
        self.queue = asyncio.Queue()
        self.total_uploads_expected = 0
        self.uploads_completed = 0
        self.finalize_received = False
        self.is_active = False

    async def run(self) -> None:
        """
        Runs the manager loop, listening for updates and emitting status to the UI.
        This should be started as a background task using asyncio.create_task().
        """
        while not (self.finalize_received and self.total_uploads_expected == self.uploads_completed):
            msg = await self.queue.get()
            msg_type = msg[0]

            if msg_type == 'REGISTER_UPLOAD':
                self.is_active = True
                self.total_uploads_expected += 1
                await self._emit_progress_update()
            elif msg_type == 'COMPLETE_UPLOAD':
                self.uploads_completed += 1
                await self._emit_progress_update()
            elif msg_type == 'FINALIZE':
                self.finalize_received = True

            self.queue.task_done()

        log.debug('UploadStatusManager finished its run.')

    async def _emit_progress_update(self) -> None:
        """Emits the current progress to the front-end if uploads are active."""
        if not self.is_active:
            return

        is_done = self.total_uploads_expected > 0 and self.uploads_completed == self.total_uploads_expected

        if is_done:
            message = f'Upload complete. {self.uploads_completed} file(s) processed.'
        else:
            # Show "Uploading 1 of N..."
            message = f'Uploading file {self.uploads_completed + 1} of {self.total_uploads_expected}...'

        self.event_emitter.emit_status(message, done=is_done, indent_level=1)


class FilesAPIManager:
    """
    Manages uploading, caching, and retrieving files using the Google Gemini Files API.

    This class provides a stateless and efficient way to handle files by using a fast,
    non-cryptographic hash (xxHash) of the file's content as the primary identifier.
    This enables content-addressable storage, preventing duplicate uploads of the
    same file. It uses a multi-tiered approach:

    1. Hot Path (In-Memory Caches): For instantly retrieving file objects and hashes
       for recently used files.
    2. Warm Path (Stateless GET): For quickly recovering file state after a server
       restart by using a deterministic name (derived from the content hash) and a
       single `get` API call.
    3. Cold Path (Upload): As a last resort, for uploading new files or re-uploading
       expired ones.
    """

    def __init__(
        self,
        client: genai.Client,
        file_cache: SimpleMemoryCache,
        id_hash_cache: SimpleMemoryCache,
        event_emitter: EventEmitter,
    ):
        """
        Initializes the FilesAPIManager.

        Args:
            client: An initialized `google.genai.Client` instance.
            file_cache: An aiocache instance for mapping `content_hash -> types.File`.
                        Must be configured with `aiocache.serializers.NullSerializer`.
            id_hash_cache: An aiocache instance for mapping `owui_file_id -> content_hash`.
                           This is an optimization to avoid re-hashing known files.
            event_emitter: An abstract class for emitting events to the front-end.
        """
        self.client = client
        self.file_cache = file_cache
        self.id_hash_cache = id_hash_cache
        self.event_emitter = event_emitter
        # A dictionary to manage locks for concurrent uploads.
        # The key is a composite of api_key_hash and content_hash.
        self.upload_locks: dict[str, asyncio.Lock] = {}
        self.api_key_hash = self._get_api_key_hash()

    def _get_api_key_hash(self) -> str:
        """
        Returns a hash of the API key for use in cache keys.

        Returns 'no_key' if the client is not using an API key (e.g., Vertex AI with ADC).
        """
        # The genai.Client object doesn't expose the API key directly.
        # It's stored in the internal _api_client.
        api_key = getattr(self.client._api_client, 'api_key', None)
        if not api_key:
            # This could happen if using Vertex AI with Application Default Credentials
            return 'no_key'
        return xxhash.xxh64(api_key.encode('utf-8')).hexdigest()

    def _get_file_cache_key(self, content_hash: str) -> str:
        """Gets the namespaced key for the file cache."""
        return f'{self.api_key_hash}:{content_hash}'

    def _get_lock_key(self, content_hash: str) -> str:
        """Gets the namespaced key for upload locks."""
        # Although the deterministic_name is content-based, the file's ownership
        # is tied to the API key (project). Locking per API key + content hash
        # allows concurrent uploads of the same file for different users.
        return f'{self.api_key_hash}:{content_hash}'

    async def get_or_upload_file(
        self,
        file_bytes: bytes,
        mime_type: str,
        *,
        owui_file_id: str | None = None,
        status_queue: asyncio.Queue | None = None,
    ) -> types.File:
        """
        The main public method to get a file, using caching, recovery, or uploading.

        This method uses a fast content hash (xxHash) as the primary key for all
        caching and remote API interactions to ensure deduplication and performance.
        It is safe from race conditions during concurrent uploads.

        Args:
            file_bytes: The raw byte content of the file. Required.
            mime_type: The MIME type of the file (e.g., 'image/png'). Required.
            owui_file_id: The unique ID of the file from Open WebUI, if available.
                      RECOMMENDED_COMPANION_VERSION    Used for logging and as a key for the hash cache optimization.
            status_queue: An optional asyncio.Queue to report upload lifecycle events.

        Returns:
            An `ACTIVE` `google.genai.types.File` object.

        Raises:
            FilesAPIError: If the file fails to upload or process.
        """
        # Step 1: Get the fast content hash, using the ID cache as an optimization if possible.
        content_hash = await self._get_content_hash(file_bytes, owui_file_id)

        # Step 2: The Hot Path (Check Local File Cache)
        # A cache hit means the file is valid and we can return immediately.
        file_cache_key = self._get_file_cache_key(content_hash)
        cached_file: types.File | None = await self.file_cache.get(file_cache_key)
        if cached_file:
            log_id = f'OWUI ID: {owui_file_id}' if owui_file_id else 'anonymous file'
            log.debug(f'Cache HIT for file hash {content_hash} ({log_id}). Returning immediately.')
            return cached_file

        # On cache miss, acquire a lock specific to this file's content to prevent race conditions.
        # dict.setdefault is atomic, ensuring only one lock is created per hash.
        lock_key = self._get_lock_key(content_hash)
        lock = self.upload_locks.setdefault(lock_key, asyncio.Lock())
        if lock.locked():
            log.debug(
                f'Lock for key {lock_key} is held by another task. This call will now wait for the lock to be released.'
            )

        async with lock:
            # Step 2.5: Double-Checked Locking
            # After acquiring the lock, check the cache again. Another task might have
            # completed the upload while we were waiting for the lock.
            cached_file = await self.file_cache.get(file_cache_key)
            if cached_file:
                log.debug(f'Cache HIT for file hash {content_hash} after acquiring lock. Returning.')
                return cached_file

            # Step 3: The Warm/Cold Path (On Cache Miss)
            # The file ID (name after "files/") must be <= 40 chars.
            # "owui-" (5) + hash (16) + "-" (1) + hash (16) = 38 chars.
            deterministic_name = f'files/owui-{self.api_key_hash}-{content_hash}'
            log.debug(
                f'Cache MISS for hash {content_hash}. Attempting stateless recovery with GET: {deterministic_name}'
            )

            try:
                # Attempt to get the file (Warm Path)
                file = await self.client.aio.files.get(name=deterministic_name)
                if not file.name:
                    raise FilesAPIError(f'Stateless recovery for {deterministic_name} returned a file without a name.')

                log.debug(f'Stateless recovery successful for {deterministic_name}. File exists on server.')
                active_file = await self._poll_for_active_state(file.name, owui_file_id)

                ttl_seconds = self._calculate_ttl(active_file.expiration_time)
                await self.file_cache.set(file_cache_key, active_file, ttl=ttl_seconds)

                return active_file
            except genai_errors.ClientError as e:
                # NOTE: The Gemini Files API returns 403 Forbidden when trying to GET
                # a file that either does not exist or belongs to another project.
                # We treat 403 as the "not found" signal for our warm path and
                # include 404 for forward compatibility.
                if e.code == 403 or e.code == 404:
                    log.info(
                        f'File {deterministic_name} not found on server (received {e.code}). Proceeding to upload.'
                    )
                    # Proceed to upload (Cold Path)
                    return await self._upload_and_process_file(
                        content_hash,
                        file_bytes,
                        mime_type,
                        deterministic_name,
                        owui_file_id,
                        status_queue,
                    )
                else:
                    log.exception(
                        f'An unhandled client error (code: {e.code}) occurred during stateless '
                        f'recovery for {deterministic_name}.'
                    )
                    self.event_emitter.emit_toast(
                        f'API error for file: {e.code}. Please check permissions.',
                        'error',
                    )
                    raise FilesAPIError(f'Failed to check file status for {deterministic_name}: {e}') from e
            except Exception as e:
                log.exception(f'An unexpected error occurred during stateless recovery for {deterministic_name}.')
                self.event_emitter.emit_toast(
                    'Unexpected error retrieving a file. Please try again.',
                    'error',
                )
                raise FilesAPIError(f'Failed to check file status for {deterministic_name}: {e}') from e
            finally:
                # Clean up the lock from the dictionary once processing is complete
                # for this hash, preventing memory growth over time.
                # This is safe because any future request for this hash will hit the cache.
                if lock_key in self.upload_locks:
                    del self.upload_locks[lock_key]

    async def _get_content_hash(self, file_bytes: bytes, owui_file_id: str | None) -> str:
        """
        Retrieves the file's content hash, using a cache for known IDs or computing it.

        This acts as a memoization layer for the hashing process, avoiding
        re-computation for files with a known Open WebUI ID. For anonymous files
        (owui_file_id=None), it will always compute the hash.
        """
        if owui_file_id:
            # First, check the ID-to-Hash cache for known files.
            # This cache is NOT namespaced by API key, as the mapping from
            # an OWUI file ID to its content hash is constant.
            cached_hash: str | None = await self.id_hash_cache.get(owui_file_id)
            if cached_hash:
                log.trace(f'Hash cache HIT for OWUI ID {owui_file_id}.')
                return cached_hash

        # If not in cache or if file is anonymous, compute the fast hash.
        log.trace(f'Hash cache MISS for OWUI ID {owui_file_id if owui_file_id else "N/A"}. Computing hash.')
        content_hash = xxhash.xxh64(file_bytes).hexdigest()

        # If there was an ID, store the newly computed hash for next time.
        if owui_file_id:
            await self.id_hash_cache.set(owui_file_id, content_hash)

        return content_hash

    def _calculate_ttl(self, expiration_time: dt.datetime | None) -> float | None:
        """Calculates the TTL in seconds from an expiration datetime."""
        if not expiration_time:
            return None

        now_utc = dt.datetime.now(dt.UTC)
        if expiration_time <= now_utc:
            return 0

        return (expiration_time - now_utc).total_seconds()

    async def _upload_and_process_file(
        self,
        content_hash: str,
        file_bytes: bytes,
        mime_type: str,
        deterministic_name: str,
        owui_file_id: str | None,
        status_queue: asyncio.Queue | None = None,
    ) -> types.File:
        """Handles the full upload and post-upload processing workflow."""

        # Register with the manager that an actual upload is starting.
        if status_queue:
            await status_queue.put(('REGISTER_UPLOAD',))

        log.info(f'Starting upload for {deterministic_name}...')

        try:
            file_io = io.BytesIO(file_bytes)
            upload_config = types.UploadFileConfig(name=deterministic_name, mime_type=mime_type)
            uploaded_file = await self.client.aio.files.upload(file=file_io, config=upload_config)
            if not uploaded_file.name:
                raise FilesAPIError(f'File upload for {deterministic_name} did not return a file name.')

            log.debug(f'{uploaded_file.name} uploaded.')
            log.trace('Uploaded file details:', payload=uploaded_file)

            # Check if the file is already active. If so, we can skip polling.
            if uploaded_file.state == types.FileState.ACTIVE:
                log.debug(f'File {uploaded_file.name} is already ACTIVE. Skipping poll.')
                active_file = uploaded_file
            else:
                # If not active, proceed with the original polling logic.
                log.debug(f'{uploaded_file.name} uploaded with state {uploaded_file.state}. Polling for ACTIVE state.')
                active_file = await self._poll_for_active_state(uploaded_file.name, owui_file_id)
                log.debug(f'File {active_file.name} is now ACTIVE.')

            # Calculate TTL and set in the main file cache using the content hash as the key.
            ttl_seconds = self._calculate_ttl(active_file.expiration_time)
            file_cache_key = self._get_file_cache_key(content_hash)
            await self.file_cache.set(file_cache_key, active_file, ttl=ttl_seconds)
            log.debug(f'Cached new file object for hash {content_hash} with TTL: {ttl_seconds}s.')

            return active_file
        except Exception as e:
            log.exception(f'File upload or processing failed for {deterministic_name}.')
            self.event_emitter.emit_toast(
                'Upload failed for a file. Please check connection and try again.',
                'error',
            )
            raise FilesAPIError(f'Upload failed for {deterministic_name}: {e}') from e
        finally:
            # Report completion (success or failure) to the status manager.
            # This ensures the progress counter always advances.
            if status_queue:
                await status_queue.put(('COMPLETE_UPLOAD',))

    async def _poll_for_active_state(
        self,
        file_name: str,
        owui_file_id: str | None,
        timeout: int = 60,
        poll_interval: int = 1,
    ) -> types.File:
        """Polls the file's status until it is ACTIVE or fails."""
        end_time = time.monotonic() + timeout
        while time.monotonic() < end_time:
            try:
                file = await self.client.aio.files.get(name=file_name)
            except Exception as e:
                raise FilesAPIError(f'Polling failed: Could not get status for {file_name}. Reason: {e}') from e

            if file.state == types.FileState.ACTIVE:
                return file
            if file.state == types.FileState.FAILED:
                log_id = f"'{owui_file_id}'" if owui_file_id else 'an uploaded file'
                error_message = f'File processing failed on server for {file_name}.'
                toast_message = f'Google could not process {log_id}.'
                if file.error:
                    reason = f'Reason: {file.error.message} (Code: {file.error.code})'
                    error_message += f' {reason}'
                    toast_message += f' Reason: {file.error.message}'

                self.event_emitter.emit_toast(toast_message, 'error')
                raise FilesAPIError(error_message)

            state_name = file.state.name if file.state else 'UNKNOWN'
            log.trace(f'File {file_name} is still {state_name}. Waiting {poll_interval}s...')
            await asyncio.sleep(poll_interval)

        raise FilesAPIError(f'File {file_name} did not become ACTIVE within {timeout} seconds.')


class GeminiContentBuilder:
    """Builds a list of `google.genai.types.Content` objects from the OWUI's body payload.

    Assistant turns are restored from the `reasoning_details` that this pipe attached to its own responses
    (exact parts and thought signatures), tool rounds become function calls followed by their function
    responses, and the files of user messages are read from the chat history in the database.
    """

    def __init__(
        self,
        messages_body: list[Message],
        metadata_body: Metadata,
        user_data: UserData,
        event_emitter: EventEmitter,
        options: ResolvedOptions,
        files_api_manager: FilesAPIManager,
        *,
        model_id: str = '',
        replay_server_side_parts: bool = False,
        max_images: int | None = None,
        image_media_resolution: types.PartMediaResolution | None = None,
    ):
        self.messages_body = messages_body
        self.upload_documents = (metadata_body.get('features', {}) or {}).get('upload_documents', False)
        # Identify if this is a background task (title/tags/etc) to optimize context.
        self.is_task = bool(metadata_body.get('task'))
        self.event_emitter = event_emitter
        self.options = options
        self.files_api_manager = files_api_manager
        self.model_id = model_id
        # `toolCall`/`toolResponse` parts are only valid while the request circulates tool context.
        self.replay_server_side_parts = replay_server_side_parts
        self.max_images = max_images
        self.image_media_resolution = image_media_resolution

        chat_id = metadata_body.get('chat_id')
        chat_id_str = chat_id if isinstance(chat_id, str) else ''
        self.is_temp_chat = not chat_id_str or 'local' in chat_id_str or 'temporary' in chat_id_str
        self.vertexai = self.files_api_manager.client.vertexai

        self.system_prompt, self.messages_body = self._extract_system_prompt(self.messages_body)
        self.metadata_body = metadata_body
        self.user_data = user_data
        self.messages_db = None
        self.user_files: dict[int, list[FileAttachmentTD]] = {}
        self._user_model: Any = None

    @cached_property
    def strict_signatures(self) -> bool:
        """Gemini 3 and later reject function calls of the current turn that lack their thought signature."""
        return bool(self.model_id) and ModelCatalog.generation(self.model_id) >= 3

    async def build_contents(self) -> list[types.Content]:
        """
        The main public method to generate the contents list by processing all
        message turns concurrently and using a self-configuring status manager.
        """
        # Fetch chat history and cumulative usage from the DB (async APIs).
        self.messages_db = await self._fetch_chat_history(self.metadata_body, self.user_data)
        log.trace('Database messages: ', payload=self.messages_db)
        self.user_files = self._align_user_files(self.messages_db)
        # Retrieve cumulative usage from the DB history and inject it into metadata.
        # This will be picked up later when constructing the final usage payload.
        c_tokens, c_cost = self._retrieve_previous_usage_data()
        self.metadata_body['cumulative_tokens'] = c_tokens
        self.metadata_body['cumulative_cost'] = c_cost

        # 1. Set up and launch the status manager. It will activate itself if needed.
        status_manager = UploadStatusManager(self.event_emitter)
        manager_task = asyncio.create_task(status_manager.run())

        # 2. Create and run concurrent processing tasks for each turn (a tool round counts as one).
        units = self._group_turns()
        tasks = [self._process_turn(unit, status_manager.queue) for unit in units]
        log.debug(f'Starting concurrent processing of {len(tasks)} message turns.')
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Signal to the manager that no more uploads will be registered.
        await status_manager.queue.put(('FINALIZE',))

        # 4. Wait for the manager to finish processing all reported uploads.
        await manager_task

        # 5. Filter and assemble the final contents list.
        contents: list[types.Content] = []
        for i, res in enumerate(results):
            if isinstance(res, list):
                contents.extend(res)
            elif isinstance(res, Exception):
                log.error(
                    f'An error occurred while processing message {i} concurrently.',
                    payload=res,
                )
        self._limit_images(contents)
        self._end_with_user_turn(contents)
        return contents

    def _retrieve_previous_usage_data(self) -> tuple[int | None, float | None]:
        """
        Retrieves the cumulative token count and cost from the last assistant message in the database.

        Returns:
            - (0, 0.0) if it's the start of a conversation (no previous assistant message).
            - (tokens, cost) if the previous assistant message has valid cumulative data.
            - (None, None) if the chain is broken (previous message exists but lacks data)
              or if DB history is unavailable (e.g., temp chat).
        """
        if not self.messages_db:
            return None, None

        for msg in reversed(self.messages_db):
            if msg.get('role') == 'assistant':
                usage = msg.get('usage', {})
                # These keys must be populated by the plugin in previous turns
                c_tokens = usage.get('cumulative_token_count')
                c_cost = usage.get('cumulative_total_cost')

                if c_tokens is not None and c_cost is not None:
                    return c_tokens, c_cost
                else:
                    # Previous assistant message exists but lacks cumulative data.
                    # This indicates a broken chain (old message or different plugin).
                    return None, None

        # No assistant message found in history, implying this is the first turn.
        return 0, 0.0

    @staticmethod
    def _extract_system_prompt(
        messages: list[Message],
    ) -> tuple[str | None, list[Message]]:
        """Extracts the system prompt and returns it along with the modified message list."""
        system_message, remaining_messages = pop_system_message(messages)  # type: ignore
        system_prompt: str | None = (system_message or {}).get('content')
        return system_prompt, remaining_messages  # type: ignore

    async def _fetch_chat_history(self, metadata_body: Metadata, user_data: UserData) -> list[ChatMessageTD] | None:
        """
        Reconstructs the active chat branch from history and removes the trailing
        assistant placeholder. Background tasks and temporary chats have no usable history.

        Every fallback to the active memory payload is logged and surfaced to the
        user via a toast with a reason-specific message.
        """
        if self.is_task:
            # Task prompts (titles, tags, summaries, ...) are built by Open WebUI from the chat itself.
            return None

        if self.is_temp_chat:
            # Expected path, not a failure: temp chats have no DB history by design.
            _log_and_toast(
                self.event_emitter,
                'Temporary chat detected; skipping database history fetch and using the active memory payload.',
                'info',
            )
            return None

        chat_id = metadata_body.get('chat_id')
        if not chat_id:
            _log_and_toast(
                self.event_emitter,
                'No chat_id in metadata; cannot fetch database history. Using the active memory payload.',
                'warning',
            )
            return None

        # Only the owner's chats: the files of messages are read with the requesting user's access rights.
        chat = await Chats.get_chat_by_id_and_user_id(id=chat_id, user_id=user_data['id'])
        if not chat:
            _log_and_toast(
                self.event_emitter,
                f'Chat {chat_id} was not found in the database; using the active memory payload.',
            )
            return None

        chat_content: ChatObjectDataTD = chat.chat  # type: ignore
        history_data = chat_content.get('history', {})
        messages_dict = history_data.get('messages', {})
        # The message being generated identifies the branch; the chat's currentId may point elsewhere.
        current_id = metadata_body.get('message_id')
        if current_id not in messages_dict:
            current_id = history_data.get('currentId')

        if not messages_dict or not current_id:
            _log_and_toast(
                self.event_emitter,
                'Chat history is empty or lacks a currentId; using the active memory payload.',
                'warning',
            )
            return None

        # 1. Walk up the parentId chain to reconstruct the linear conversation branch.
        messages_db: list[ChatMessageTD] = []
        curr_id = current_id
        visited: set[str] = set()

        while curr_id and curr_id in messages_dict and curr_id not in visited:
            visited.add(curr_id)
            msg = messages_dict[curr_id]
            messages_db.insert(0, msg)
            curr_id = msg.get('parentId')

        # 2. Handle the trailing assistant placeholder.
        # OWUI inserts an assistant message entry for the turn currently being processed.
        if messages_db and messages_db[-1].get('role') == 'assistant':
            messages_db.pop()

        return messages_db

    def _align_user_files(self, messages_db: list[ChatMessageTD] | None) -> dict[int, list[FileAttachmentTD]]:
        """
        Maps the index of each user message in the request body to the files of its database message.

        The body is longer than the history when it holds tool rounds and shorter when Open WebUI compacted
        the context, so user messages are paired from the end. Messages that Open WebUI adds itself (the
        images of tool results, the guided-regeneration prompt) have no database counterpart.
        """
        if not messages_db:
            return {}

        db_users = [message for message in messages_db if message.get('role') == 'user']
        body_users = [
            i
            for i, message in enumerate(self.messages_body)
            if message.get('role') == 'user' and not self._is_tool_images_message(message)
        ]
        if self.metadata_body.get('regeneration_message_id') and body_users:
            body_users.pop()

        return {i: message.get('files') or [] for i, message in zip(reversed(body_users), reversed(db_users))}

    def _group_turns(self) -> list[tuple[str, int, list[Message]]]:
        """
        Groups the body messages into turns: ('user' | 'assistant', index, [message]) or
        ('tool_round', index, [assistant message with tool_calls, tool messages...]).

        A tool round whose assistant message is missing keeps only its tool messages, which are dropped later.
        """
        units: list[tuple[str, int, list[Message]]] = []
        i = 0
        while i < len(self.messages_body):
            message = self.messages_body[i]
            role = message.get('role')
            if (role == 'assistant' and message.get('tool_calls')) or role == 'tool':
                end = i + 1 if role == 'assistant' else i
                while end < len(self.messages_body) and self.messages_body[end].get('role') == 'tool':
                    end += 1
                units.append(('tool_round', i, list(self.messages_body[i:end])))
                i = end
                continue
            units.append((cast(str, role), i, [message]))
            i += 1
        return units

    def _current_turn_start(self) -> int:
        """Index of the newest user message: Gemini validates the function calls after it."""
        return max((i for i, m in enumerate(self.messages_body) if m.get('role') == 'user'), default=-1)

    async def _process_turn(
        self, unit: tuple[str, int, list[Message]], status_queue: asyncio.Queue
    ) -> list[types.Content]:
        """
        Processes one turn and returns its `types.Content` objects. Designed to be run concurrently.
        """
        kind, i, messages = unit
        if kind == 'tool_round':
            return await self._process_tool_round(i, messages, i > self._current_turn_start(), status_queue)

        message = messages[0]
        parts: list[types.Part] = []
        role = kind

        if role == 'user':
            message = cast('UserMessage', message)
            # Logic for retrieving files is now handled inside _process_user_message
            # to allow for finer control over which file types are included based on task mode.
            parts = await self._process_user_message(i, message, status_queue)
            # Case 1: User content is completely empty (no text, no files).
            if not parts:
                _log_and_toast(
                    self.event_emitter,
                    f'Your message #{i + 1} was completely empty. The assistant will ask for clarification.',
                    'warning',
                )
                clarification_prompt = (
                    'The user sent an empty message. Please ask the user for '
                    'clarification on what they would like to ask or discuss.'
                )
                # This will become the only part for this user message.
                parts = await self._genai_parts_from_text(clarification_prompt, status_queue)
            else:
                # Case 2: User has sent content, check if it includes text.
                has_text_component = any(p.text for p in parts if p.text)
                if not has_text_component and self.vertexai:
                    # Vertex AI requires a text part in multi-modal messages.
                    _log_and_toast(
                        self.event_emitter,
                        f'For your message #{i + 1}, a default prompt was added as text is required '
                        'for requests with attachments when using Vertex AI.',
                        'warning',
                    )
                    parts.extend(await self._genai_parts_from_text(ATTACHMENTS_ONLY_PROMPT, status_queue))
                elif not has_text_component:
                    # Google Developer API allows no-text user content (except in the last turn, see
                    # `_end_with_user_turn`).
                    log.debug(f'User message at index {i} lacks a text component. Proceeding with non-text parts.')
        elif role == 'assistant':
            message = cast('AssistantMessage', message)
            # Google API's assistant role is "model"
            role = 'model'
            parts = await self._process_assistant_message(i, message, status_queue)
        else:
            _log_and_toast(
                self.event_emitter,
                f'Message {i} has an invalid role: {role}. Skipping to the next message.',
                'warning',
            )
            return []

        # Only create a Content object if there are parts to include.
        if parts:
            return [types.Content(parts=parts, role=role)]
        return []

    async def _process_user_message(
        self,
        i: int,
        message: UserMessage,
        status_queue: asyncio.Queue,
    ) -> list[types.Part]:
        user_parts: list[types.Part] = []
        db_files_processed = False

        # PATH 1: Database is available (Normal Chat).
        if files := self.user_files.get(i):
            db_files_processed = True
            upload_tasks = []

            for file in files:
                content_type = file.get('content_type', '')
                # MIME types for images always start with 'image/' (e.g., image/png, image/jpeg)
                is_image = content_type.startswith('image/') or file.get('type') == 'image'

                # Optimization: Task models (titles, tags, etc.) skip heavy documents
                # but keep images as they provide high context value for low token cost.
                should_include = is_image or (self.upload_documents and not self.is_task)

                if not should_include:
                    log.debug(
                        f"Skipping {content_type} '{file.get('id')}' "
                        f'(is_task={self.is_task}, upload_documents={self.upload_documents})'
                    )
                    continue

                # We always use the internal API endpoint to fetch file content from the DB.
                # Even for images, the 'url' field in the attachment object is often
                # just the UUID, which isn't fetchable on its own.
                if file_id := file.get('id'):
                    uri = f'/api/v1/files/{file_id}/content'
                    upload_tasks.append(self._genai_part_from_uri(uri, status_queue))
                else:
                    _log_and_toast(
                        self.event_emitter,
                        f'Encountered a malformed file object in message #{i + 1} '
                        "without an ID; it will not be injected into the model's context.",
                        'warning',
                    )

            if upload_tasks:
                log.info(f'Processing {len(upload_tasks)} file(s) from database.')
                results = await asyncio.gather(*upload_tasks)
                user_parts.extend(part for part in results if part)

        # Now, process the content from the message payload.
        user_content = message.get('content')
        if isinstance(user_content, str):
            user_content_list: list[Content] = [{'type': 'text', 'text': user_content}]
        elif isinstance(user_content, list):
            user_content_list = user_content
        else:
            _log_and_toast(
                self.event_emitter,
                f'Message #{i + 1} has invalid content (not a string or list); skipping the malformed content.',
                'warning',
            )
            return user_parts

        for c in user_content_list:
            c_type = c.get('type')
            if c_type == 'text':
                c = cast('TextContent', c)
                if c_text := c.get('text'):
                    user_parts.extend(await self._genai_parts_from_text(c_text, status_queue))

            # PATH 2: Temporary Chat Image Handling.
            # FIXME: this puts images to the end of the message, see if it matters where they are.
            elif c_type == 'image_url' and not db_files_processed:
                log.info('Processing image from payload (temporary chat mode).')
                c = cast('ImageContent', c)
                if uri := c.get('image_url', {}).get('url'):
                    if part := await self._genai_part_from_uri(uri, status_queue):
                        user_parts.append(part)

        return user_parts

    async def _rehydrate_assistant_parts(
        self,
        gemini_parts: list[dict[str, Any]],
        status_queue: asyncio.Queue,
    ) -> list[types.Part]:
        """
        Reconstructs `types.Part` objects from dictionaries, rehydrating file-based parts
        by fetching their content from the OWUI backend.
        """
        rehydrated_parts: list[types.Part] = []
        for part_dict in gemini_parts:
            part = types.Part.model_validate(part_dict)

            if part.file_data and (file_uri := part.file_data.file_uri):
                if not file_uri.startswith('/api/v1/files/'):
                    raise ValueError(
                        f'Unsupported file_uri in assistant history: {file_uri}. '
                        'Only local Open WebUI files are supported for reconstruction.'
                    )

                file_id = file_uri.split('/')[4]
                file_bytes, mime_type = await self._get_file_data(file_id)

                if not (file_bytes and mime_type):
                    raise ValueError(f"Could not retrieve content for file_id '{file_id}' from assistant history.")

                # Force raw bytes (inline_data) to preserve exact history context for the model.
                # This ensures we don't convert original inline images into Files API references.
                rehydrated_part = await self._create_genai_part_from_file_data(
                    file_bytes, mime_type, file_id, status_queue, force_raw=True
                )
                part.inline_data = rehydrated_part.inline_data
                part.file_data = rehydrated_part.file_data
                rehydrated_parts.append(part)
            else:
                rehydrated_parts.append(part)

        return rehydrated_parts

    def _pop_thoughts(self, content: str) -> tuple[str, list[str]]:
        """
        Identifies and removes thought blocks from the content.

        A thought is defined as text between <think> and </think>\n.
        This method handles multiple thought blocks if they are peppered
        throughout the message.

        :param content: The raw message content from the assistant.
        :return: A tuple containing (cleaned_content, list_of_extracted_thoughts).
        """
        # The pattern looks for the <​think> tag, captures everything inside (non-greedy),
        # and matches the <​/think> tag plus a potential trailing newline.
        # re.DOTALL allows the '.' character to match newlines within the capture group.
        thought_pattern = re.compile(r'<think>(.*?)</think>\n?', re.DOTALL)

        thoughts = thought_pattern.findall(content)
        # Replace all occurrences with an empty string to get the "clean" content.
        cleaned_content = thought_pattern.sub('', content)

        return cleaned_content, thoughts

    async def _process_assistant_message(
        self,
        i: int,
        message_body: AssistantMessage,
        status_queue: asyncio.Queue,
    ) -> list[types.Part]:
        """
        Processes an assistant message, preferring the exact parts this pipe stored in `reasoning_details`
        when the visible text is unedited. Falls back to processing the text content otherwise.
        """
        if (parts := await self._replayed_parts(i, message_body, status_queue)) is not None:
            return parts

        # Fallback to processing text content: messages of other models, edited messages and messages that
        # did not need stored parts (plain text without thought signatures).
        current_content, _ = self._pop_thoughts(self._message_text(message_body))
        log.debug(f'Processing assistant message {i} content as plain text.')
        return await self._genai_parts_from_text(current_content, status_queue)

    async def _replayed_parts(
        self,
        i: int,
        message: AssistantMessage,
        status_queue: asyncio.Queue,
    ) -> list[types.Part] | None:
        """
        Rebuilds the model parts stored in the message's `reasoning_details`, or returns None if there are
        none, they do not match the visible text (the user edited the message) or they cannot be restored.
        """
        part_dicts = self._stored_part_dicts(message)
        if part_dicts is None:
            return None

        visible = self._message_text(message)
        stored_text = ''.join(self._displayed_text(p) for p in part_dicts)
        if _comparable_text(stored_text) != _comparable_text(visible):
            # A meaningful edit was detected after accounting for whitespace, citation markers and attributions.
            _log_and_toast(
                self.event_emitter,
                f'An edit was detected in assistant message #{i + 1}. '
                'Using the edited text, which may affect model context for this turn.',
                'warning',
            )
            diff = difflib.unified_diff(
                stored_text.strip().splitlines(keepends=True),
                visible.strip().splitlines(keepends=True),
                fromfile='original_content',
                tofile='current_content',
            )
            log.warning(f'Edited content diff for assistant message {i}:\n{"".join(diff)}')
            return None

        if not self.replay_server_side_parts:
            # Without tool context circulation in this request, Gemini would reject server-side tool parts.
            part_dicts = [p for p in part_dicts if p.get('tool_call') is None and p.get('tool_response') is None]

        log.debug(f'Reconstructing assistant message at index {i} from stored parts.')
        try:
            return await self._rehydrate_assistant_parts(part_dicts, status_queue)
        except (pydantic_core.ValidationError, TypeError, ValueError) as exc:
            _log_and_toast(
                self.event_emitter,
                f'Failed to reconstruct assistant message #{i + 1} from stored data: {exc}. '
                'Falling back to plain text processing.',
                'warning',
            )
            return None

    @staticmethod
    def _displayed_text(part: dict[str, Any]) -> str:
        """What the pipe showed for a stored part: its text, the link of a generated image, or a code block."""
        if part.get('thought'):
            return ''
        if isinstance(text := part.get('text'), str):
            return text
        file_data = part.get('file_data') or {}
        if str(file_data.get('mime_type') or '').startswith('image/') and file_data.get('file_uri'):
            return _image_markdown(file_data['file_uri'])
        if code := part.get('executable_code'):
            return Pipe._process_executable_code_part(types.ExecutableCode.model_validate(code)) or ''
        if result := part.get('code_execution_result'):
            return Pipe._process_code_execution_result_part(types.CodeExecutionResult.model_validate(result)) or ''
        return ''

    @staticmethod
    def _stored_part_dicts(message: Message) -> list[dict[str, Any]] | None:
        """The parts of the rounds this pipe stored in the message's `reasoning_details` (REPLAY_FORMAT)."""
        details = message.get('reasoning_details')
        if not details:
            fields = message.get('provider_specific_fields')
            details = fields.get('reasoning_details') if isinstance(fields, dict) else None
        if not isinstance(details, list):
            return None

        part_dicts: list[dict[str, Any]] = []
        found = False
        for item in details:
            if not isinstance(item, dict) or item.get('format') != REPLAY_FORMAT:
                continue
            found = True
            try:
                parts = json.loads(item.get('data') or '').get('parts') or []
            except (json.JSONDecodeError, AttributeError, TypeError) as exc:
                log.warning(f'Ignoring unreadable stored model content: {exc}')
                return None
            part_dicts.extend(p for p in parts if isinstance(p, dict))
        return part_dicts if found else None

    @staticmethod
    def _message_text(message: Message) -> str:
        """The text of a message whose content is a string or a list of content parts."""
        content = message.get('content')
        if isinstance(content, list):
            return ''.join(
                c.get('text') or '' for c in content if isinstance(c, dict) and c.get('type') in ('text', 'output_text')
            )
        return content if isinstance(content, str) else ''

    @staticmethod
    def _is_tool_images_message(message: Message) -> bool:
        """Whether a message is the user message in which Open WebUI passes on the images of tool results."""
        content = message.get('content')
        return (
            message.get('role') == 'user'
            and isinstance(content, list)
            and bool(content)
            and isinstance(content[0], dict)
            and content[0].get('text') == OWUI_TOOL_IMAGES_TEXT
        )

    async def _process_tool_round(
        self,
        i: int,
        messages: list[Message],
        current_turn: bool,
        status_queue: asyncio.Queue,
    ) -> list[types.Content]:
        """
        Converts an assistant message with tool_calls and the tool messages after it into a model content
        with the function calls followed by one user content with their function responses (FC1, FC2, FR1,
        FR2). Calls without a result and results without a call are dropped: Gemini rejects both.
        """
        assistant = messages[0] if messages[0].get('role') == 'assistant' else None
        tool_messages = [m for m in messages if m.get('role') == 'tool']
        if assistant is None:
            log.warning(f'Dropping {len(tool_messages)} tool result(s) at index {i} that follow no tool call.')
            return []

        tool_calls = [tc for tc in assistant.get('tool_calls') or [] if isinstance(tc, dict)]
        answered = {str(m.get('tool_call_id')) for m in tool_messages if m.get('tool_call_id')}

        parts = await self._replayed_parts(i, cast('AssistantMessage', assistant), status_queue)
        if parts is not None and not self._calls_match(parts, tool_calls):
            log.warning(f'Stored model content of tool round {i} does not match its tool calls; rebuilding it.')
            parts = None
        if parts is None:
            parts = await self._genai_parts_from_text(self._message_text(assistant), status_queue)
            parts.extend(self._function_call_part(tool_call) for tool_call in tool_calls)

        # The function_call parts correspond 1:1 (in order) to the tool calls.
        kept: list[types.Part] = []
        responses_for: dict[str, tuple[str, str | None]] = {}
        position = 0
        for part in parts:
            if not part.function_call:
                kept.append(part)
                continue
            call_id = str(tool_calls[position].get('id') or '') if position < len(tool_calls) else ''
            position += 1
            if call_id in answered:
                responses_for[call_id] = (part.function_call.name or '', part.function_call.id)
                kept.append(part)
            else:
                log.warning(f"Dropping function call '{part.function_call.name}' without a result.")

        contents: list[types.Content] = []
        calls = [p for p in kept if p.function_call]
        if calls and current_turn and self.strict_signatures and not calls[0].thought_signature:
            # Applied after the pairing, so a dropped first call cannot leave the step without a signature.
            # As str: the SDK decodes it like a real (base64) signature and sends the literal value back.
            first = kept.index(calls[0])
            kept[first] = types.Part.model_validate(
                {**calls[0].model_dump(exclude_none=True), 'thought_signature': SKIP_THOUGHT_SIGNATURE}
            )
        if calls or any(p.text for p in kept):
            contents.append(types.Content(role='model', parts=kept if calls else [p for p in kept if p.text]))

        responses = [
            part
            for tool_message in tool_messages
            if (part := self._function_response_part(tool_message, responses_for)) is not None
        ]
        if responses:
            contents.append(types.Content(role='user', parts=responses))
        return contents

    @staticmethod
    def _calls_match(parts: list[types.Part], tool_calls: list[dict[str, Any]]) -> bool:
        """Whether stored function call parts belong to these tool calls (same count and names, in order)."""
        calls = [p.function_call for p in parts if p.function_call]
        return len(calls) == len(tool_calls) and all(
            call.name == ToolBridge.gemini_name(str((tool_call.get('function') or {}).get('name') or ''))
            for call, tool_call in zip(calls, tool_calls)
        )

    @staticmethod
    def _function_call_part(tool_call: dict[str, Any]) -> types.Part:
        """Builds a function_call part from an OpenAI tool call (a call without stored model content)."""
        function = tool_call.get('function') if isinstance(tool_call.get('function'), dict) else {}
        name = ToolBridge.gemini_name(str(function.get('name') or ''))
        arguments = function.get('arguments')
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict):
            log.warning(f"Invalid arguments of tool call '{name}'; sending an empty object.")
            args = {}
        call_id = tool_call.get('id')
        return types.Part(function_call=types.FunctionCall(id=str(call_id) if call_id else None, name=name, args=args))

    def _function_response_part(
        self,
        tool_message: Message,
        responses_for: dict[str, tuple[str, str | None]],
    ) -> types.Part | None:
        """Builds a function_response part from a tool message, echoing the name and id of its call."""
        call_id = str(tool_message.get('tool_call_id') or '')
        if call_id not in responses_for:
            log.warning(f"Dropping tool result '{call_id}' without a matching function call.")
            return None
        name, gemini_call_id = responses_for[call_id]
        text = self._message_text(tool_message)
        is_error = text.strip().lower().startswith(('error:', 'exception:', 'traceback'))
        return types.Part(
            function_response=types.FunctionResponse(
                id=gemini_call_id, name=name, response={'error' if is_error else 'output': text}
            )
        )

    def _limit_images(self, contents: list[types.Content]) -> None:
        """Drops the oldest images when the request exceeds the model's documented reference image limit."""
        if not self.max_images:
            return
        images = [
            (content, part)
            for content in contents
            for part in content.parts or []
            if (data := part.inline_data or part.file_data) and (data.mime_type or '').startswith('image/')
        ]
        excess = len(images) - self.max_images
        if excess <= 0:
            return
        for content, part in images[:excess]:
            content.parts = [p for p in content.parts or [] if p is not part]
        _log_and_toast(
            self.event_emitter,
            f'This model accepts up to {self.max_images} images per request; the {excess} oldest were left out.',
            'warning',
        )

    def _end_with_user_turn(self, contents: list[types.Content]) -> None:
        """
        Makes the request end with a user turn that holds text: Gemini 3.5 Flash-Lite, 3.6 Flash and later
        reject a request whose last turn is a model turn, and Gemini 3.8+ expects text in the last user turn.
        """
        last = contents[-1] if contents else None
        if last is not None and last.role == 'model':
            log.info('The request ends with a model turn (continued answer); asking the model to continue instead.')
            contents.append(types.Content(role='user', parts=[types.Part.from_text(text=CONTINUE_PROMPT)]))
        elif last is not None and not any(p.text or p.function_response for p in last.parts or []):
            log.debug('The last user turn holds only attachments; adding a default prompt.')
            last.parts = [*(last.parts or []), types.Part.from_text(text=ATTACHMENTS_ONLY_PROMPT)]

    async def _genai_parts_from_text(self, text: str, status_queue: asyncio.Queue) -> list[types.Part]:
        if not text:
            return []

        text = self._enable_special_tags(text)
        parts: list[types.Part] = []
        last_pos = 0

        # Conditionally build a regex to find media links.
        # If YouTube parsing is disabled, the regex will only find markdown image links,
        # leaving YouTube URLs to be treated as plain text.
        markdown_part = r'!\[.*?\]\(([^)]+)\)'  # Group 1: Markdown URI
        # Group 2: YouTube URL
        youtube_part = r'(https?://(?:(?:www|music)\.)?youtube\.com/(?:watch\?v=|shorts/|live/)[^\s)]+'
        youtube_part += r'|https?://youtu\.be/[^\s)]+)'
        # (Original single regex split into two `+=` halves to fit the line limit; pattern is identical.)
        if self.options.PARSE_YOUTUBE_URLS:
            pattern = re.compile(f'{markdown_part}|{youtube_part}')
            process_youtube = True
        else:
            pattern = re.compile(markdown_part)
            process_youtube = False
            log.info('YouTube URL parsing is disabled. URLs will be treated as plain text.')

        for match in pattern.finditer(text):
            # Add the text segment that precedes the media link
            if text_segment := text[last_pos : match.start()].strip():
                parts.append(types.Part.from_text(text=text_segment))

            # The URI is in group 1 for markdown, or group 2 for YouTube.
            if process_youtube:
                uri = match.group(1) or match.group(2)
            else:
                uri = match.group(1)

            if not uri:
                _log_and_toast(
                    self.event_emitter,
                    f'Failed to extract URI from match: {match.group(0)}. Skipping.',
                    'warning',
                )
                continue

            # Delegate all URI processing to the unified helper
            if media_part := await self._genai_part_from_uri(uri, status_queue):
                parts.append(media_part)

            last_pos = match.end()

        # Add any remaining text after the last media link
        if remaining_text := text[last_pos:].strip():
            parts.append(types.Part.from_text(text=remaining_text))

        # If no media links were found, the whole text is a single part
        if not parts and text.strip():
            parts.append(types.Part.from_text(text=text.strip()))

        return parts

    async def _genai_part_from_uri(self, uri: str, status_queue: asyncio.Queue) -> types.Part | None:
        """
        Processes any resource URI and returns a genai.types.Part.
        This is the central dispatcher for all media processing, handling data URIs,
        local API file paths, and YouTube URLs.
        """
        if not uri:
            log.warning('Received an empty URI, skipping.')
            return None

        try:
            file_bytes: bytes | None = None
            mime_type: str | None = None
            owui_file_id: str | None = None

            # Step 1: Extract bytes and mime_type from the URI if applicable
            if uri.startswith('data:image'):
                match = re.match(r'data:(image/\w+);base64,(.+)', uri)
                if not match:
                    raise ValueError('Invalid data URI for image.')
                mime_type, base64_data = match.group(1), match.group(2)
                file_bytes = base64.b64decode(base64_data)
            elif uri.startswith('/api/v1/files/'):
                log.info(f'Processing local API file URI: {uri}')
                file_id = uri.split('/')[4]
                owui_file_id = file_id
                file_bytes, mime_type = await self._get_file_data(file_id)
            elif 'youtube.com/' in uri or 'youtu.be/' in uri:
                log.info(f'Found YouTube URL: {uri}')
                return self._genai_part_from_youtube_uri(uri)
            # TODO: Google Cloud Storage bucket support.
            # elif uri.startswith("gs://"): ...
            else:
                _log_and_toast(
                    self.event_emitter,
                    f"Unsupported URI: '{uri[:64]}...' Links must be to YouTube or a supported file type.",
                    'warning',
                )
                return None

            # Step 2: If we have bytes, create the Part using the modularized helper
            if file_bytes and mime_type:
                return await self._create_genai_part_from_file_data(
                    file_bytes=file_bytes,
                    mime_type=mime_type,
                    owui_file_id=owui_file_id,
                    status_queue=status_queue,
                )

            return None  # Return None if bytes/mime_type could not be determined

        except FilesAPIError as e:
            _log_and_toast(
                self.event_emitter,
                f"Files API failed for URI '{uri[:64]}[...]': {e}",
                'error',
            )
            return None
        except Exception as e:
            _log_and_toast(
                self.event_emitter,
                f'Error processing URI: {uri[:64]}[...]: {e}',
                'error',
            )
            return None

    async def _create_genai_part_from_file_data(
        self,
        file_bytes: bytes,
        mime_type: str,
        owui_file_id: str | None,
        status_queue: asyncio.Queue,
        force_raw: bool = False,
    ) -> types.Part:
        """
        Creates a `types.Part` from file bytes, deciding whether to use the
        Google Files API or send raw bytes based on configuration and context.
        """
        # TODO: The Files API is strict about MIME types (e.g., text/plain,
        # application/pdf). In the future, inspect the content of files
        # with unsupported text-like MIME types (e.g., 'application/json',
        # 'text/markdown'). If the content is detected as plaintext,
        # override the `mime_type` variable to 'text/plain' to allow the upload.

        # Determine whether to use the Files API based on the specified conditions.
        use_files_api = True
        reason = ''

        if force_raw:
            reason = 'raw bytes are forced (e.g. for assistant history reconstruction)'
            use_files_api = False
        elif not self.options.USE_FILES_API:
            reason = 'disabled by user setting (USE_FILES_API=False)'
            use_files_api = False
        elif self.vertexai:
            reason = 'the active client is configured for Vertex AI, which does not support the Files API'
            use_files_api = False
        elif self.is_temp_chat:
            reason = 'temporary chat mode is active'
            use_files_api = False

        if use_files_api:
            log.info('Using Google Files API for resource.')
            gemini_file = await self.files_api_manager.get_or_upload_file(
                file_bytes=file_bytes,
                mime_type=mime_type,
                owui_file_id=owui_file_id,
                status_queue=status_queue,
            )
            part = types.Part(
                file_data=types.FileData(
                    file_uri=gemini_file.uri,
                    mime_type=gemini_file.mime_type,
                )
            )
        else:
            log.info(f'Sending raw bytes because {reason}.')
            part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

        # Per-part resolution (Gemini 3+, v1beta) is how images get MEDIA_RESOLUTION_ULTRA_HIGH.
        if self.image_media_resolution and mime_type.startswith('image/') and not force_raw:
            part.media_resolution = self.image_media_resolution
        return part

    def _genai_part_from_youtube_uri(self, uri: str) -> types.Part | None:
        """Creates a Gemini Part from a YouTube URL, with optional video metadata.

        Handles standard (`watch?v=`), short (`youtu.be/`), mobile (`shorts/`),
        and live (`live/`) URLs. Metadata is parsed for the Gemini Developer API
        but ignored for Vertex AI, which receives a simple URI Part.

        - **Start/End Time**: `?t=<value>` and `#end=<value>`. The value can be a
          flexible duration (e.g., "1m30s", "90") and will be converted to seconds.
        - **Frame Rate**: Can be specified in two ways (if both are present,
          `interval` takes precedence):
          - **Interval**: `#interval=<value>` (e.g., `#interval=10s`, `#interval=0.5s`).
            The value is a flexible duration converted to seconds, then to FPS (1/interval).
          - **FPS**: `#fps=<value>` (e.g., `#fps=2.5`).
          The final FPS value must be in the range (0, 24].

        Args:
            uri: The raw YouTube URL from the user.
            is_vertex_client: If True, creates a simple Part for Vertex AI.

        Returns:
            A `types.Part` object, or `None` if the URI is not a valid YouTube link.
        """
        # Convert YouTube Music URLs to standard YouTube URLs for consistent parsing.
        if 'music.youtube.com' in uri:
            uri = uri.replace('music.youtube.com', 'www.youtube.com')
            log.info(f'Converted YouTube Music URL to standard URL: {uri}')

        # Regex to capture the 11-character video ID from various YouTube URL formats.
        video_id_pattern = re.compile(
            r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|shorts/|live/)|youtu.be/)([a-zA-Z0-9_-]{11})'
        )

        match = video_id_pattern.search(uri)
        if not match:
            log.warning(f'Could not extract a valid YouTube video ID from URI: {uri}')
            return None

        video_id = match.group(1)
        canonical_uri = f'https://www.youtube.com/watch?v={video_id}'

        # --- Branching logic for Vertex AI vs. Gemini Developer API ---
        if self.vertexai:
            return types.Part.from_uri(file_uri=canonical_uri, mime_type='video/mp4')
        else:
            parsed_uri = urlparse(uri)
            query_params = parse_qs(parsed_uri.query)
            fragment_params = parse_qs(parsed_uri.fragment)

            start_offset: str | None = None
            end_offset: str | None = None
            fps: float | None = None

            # Start time from query `t`. Convert flexible format to "Ns".
            if 't' in query_params:
                raw_start = query_params['t'][0]
                if (total_seconds := self._parse_duration_to_seconds(raw_start)) is not None:
                    start_offset = f'{total_seconds}s'

            # End time from fragment `end`. Convert flexible format to "Ns".
            if 'end' in fragment_params:
                raw_end = fragment_params['end'][0]
                if (total_seconds := self._parse_duration_to_seconds(raw_end)) is not None:
                    end_offset = f'{total_seconds}s'

            # Frame rate from fragment `interval` or `fps`. `interval` takes precedence.
            if 'interval' in fragment_params:
                raw_interval = fragment_params['interval'][0]
                if (
                    interval_seconds := self._parse_duration_to_seconds(raw_interval)
                ) is not None and interval_seconds > 0:
                    calculated_fps = 1.0 / interval_seconds
                    if 0.0 < calculated_fps <= 24.0:
                        fps = calculated_fps
                    else:
                        log.warning(
                            f"Interval '{raw_interval}' results in FPS '{calculated_fps}'"
                            ' which is outside the valid range (0.0, 24.0]. Ignoring.'
                        )

            # Fall back to `fps` param if not set by `interval`.
            if fps is None and 'fps' in fragment_params:
                try:
                    fps_val = float(fragment_params['fps'][0])
                    if 0.0 < fps_val <= 24.0:
                        fps = fps_val
                    else:
                        log.warning(f"FPS value '{fps_val}' is outside the valid range (0.0, 24.0]. Ignoring.")
                except (ValueError, IndexError):
                    log.warning(f'Invalid FPS value in fragment: {fragment_params.get("fps")}. Ignoring.')

            video_metadata: types.VideoMetadata | None = None
            if start_offset or end_offset or fps is not None:
                video_metadata = types.VideoMetadata(
                    start_offset=start_offset,
                    end_offset=end_offset,
                    fps=fps,
                )

            return types.Part(
                file_data=types.FileData(file_uri=canonical_uri),
                video_metadata=video_metadata,
            )

    def _parse_duration_to_seconds(self, duration_str: str) -> float | None:
        """Converts a human-readable duration string to total seconds.

        Supports formats like "1h30m15s", "90m", "3600s", or just "90".
        Also supports float values like "0.5s" or "90.5".
        Returns total seconds as a float, or None if the string is invalid.
        """
        # First, try to convert the whole string as a plain number (e.g., "90", "90.5").
        try:
            return float(duration_str)
        except ValueError:
            # If it fails, it might be a composite duration like "1m30s", so we parse it below.
            pass

        total_seconds = 0.0
        # Regex to find number-unit pairs (e.g., 1h, 30.5m, 15s). Supports floats.
        parts = re.findall(r'(\d+(?:\.\d+)?)\s*(h|m|s)?', duration_str, re.IGNORECASE)

        if not parts:
            # log.warning(f"Could not parse duration string: {duration_str}")
            return None

        for value, unit in parts:
            val = float(value)
            unit = (unit or 's').lower()  # Default to seconds if no unit
            if unit == 'h':
                total_seconds += val * 3600
            elif unit == 'm':
                total_seconds += val * 60
            elif unit == 's':
                total_seconds += val

        return total_seconds

    @staticmethod
    def _enable_special_tags(text: str) -> str:
        """
        Reverses the action of _disable_special_tags by removing the ZWS
        from special tags. This is used to clean up history messages before
        sending them to the model, so it can understand the context correctly.
        """
        if not text:
            return ''

        # The regex finds '<ZWS' followed by an optional '/' and then one of the special tags.
        # The inner parentheses group the tags, so the optional '/' applies to all of them.
        REVERSE_TAG_REGEX = re.compile(
            r'<' + ZWS + r'(/?' + '(' + '|'.join(re.escape(tag) for tag in SPECIAL_TAGS_TO_DISABLE) + ')' + r')'
        )
        # The substitution restores the original tag, e.g., '<ZWS/think' becomes '</think'.
        restored_text, count = REVERSE_TAG_REGEX.subn(r'<\1', text)
        if count > 0:
            log.debug(f'Re-enabled {count} special tag(s) for model context.')

        return restored_text

    async def _get_file_data(self, file_id: str) -> tuple[bytes | None, str | None]:
        """
        Retrieves file metadata from the database and its content from the configured storage provider,
        if the requesting user may read the file.
        """
        # TODO: Emit toasts on unexpected conditions.
        if not file_id:
            log.warning('file_id is empty. Cannot continue.')
            return None, None

        # Await the async database call directly.
        try:
            file_model = await Files.get_file_by_id(file_id)
        except Exception as e:
            log.exception(f'An unexpected error occurred during database call for file_id {file_id}: {e}')
            return None, None

        if file_model is None:
            # The get_file_by_id method already handles and logs the specific exception,
            # so we just need to handle the None return value.
            log.warning(f"File {file_id} not found in the backend's database.")
            return None, None

        if not (file_path := file_model.path):
            log.warning(f'File {file_id} was found in the database but it lacks `path` field. Cannot Continue.')
            return None, None
        if file_model.meta is None:
            log.warning(f'File {file_path} was found in the database but it lacks `meta` field. Cannot continue.')
            return None, None
        if not (content_type := file_model.meta.get('content_type')):
            log.warning(
                f'File {file_path} was found in the database but it lacks `meta.content_type` field. Cannot continue.'
            )
            return None, None
        if not await self._may_read_file(file_model):
            log.warning(f'User {self.user_data.get("id")} may not read file {file_id}; leaving it out.')
            return None, None

        try:
            # The storage provider downloads S3, GCS and Azure objects to the local cache and returns that path.
            local_path = await asyncio.to_thread(Storage.get_file, file_path)
            async with aiofiles.open(local_path, 'rb') as file:
                file_data = await file.read()
            return file_data, content_type
        except FileNotFoundError:
            log.exception(f'File {file_path} not found on disk.')
            return None, content_type
        except Exception:
            log.exception(f'Error processing file {file_path}')
            return None, content_type

    async def _may_read_file(self, file_model: Any) -> bool:
        """Open WebUI's own read check: the owner, an admin, or access through knowledge, models, chats, ..."""
        user_id = self.user_data.get('id')
        if file_model.user_id == user_id:
            return True
        if self._user_model is None:
            self._user_model = await Users.get_user_by_id(user_id) if user_id else None
        user = self._user_model
        if user is None:
            return False
        return user.role == 'admin' or await has_access_to_file(file_model.id, 'read', user)


# region Valves

# Thinking level names accepted in THINKING_CONFIG_RULES. Keep in sync with
# `types.ThinkingLevel` (minus THINKING_LEVEL_UNSPECIFIED). XHIGH needs google-genai>=3.3.0.
_THINKING_LEVELS: Final[tuple[str, ...]] = ('MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'XHIGH')
# Requests as little thinking as the model allows (OpenAI-style `reasoning_effort: none`, or the reasoning
# toggle switched off): no thinking where it can be disabled, the lowest level or budget otherwise.
_THINKING_OFF: Final[str] = 'NONE'

_VALVE_DESCRIPTIONS: Final[dict[str, str]] = {
    # Auth & API Keys
    'GEMINI_FREE_API_KEY': 'Free Gemini Developer API key.',
    'GEMINI_PAID_API_KEY': 'Paid Gemini Developer API key.',
    'USER_MUST_PROVIDE_AUTH_CONFIG': (
        'Require all users (including admins) to provide their own authentication credentials via `UserValves`.\n\n'
        'Setting this to `True` prevents non-whitelisted users from using Vertex AI.'
    ),
    'AUTH_WHITELIST': (
        'Comma-separated list of user email addresses allowed to bypass `USER_MUST_PROVIDE_AUTH_CONFIG`'
        ' and use default system credentials.'
    ),
    'GEMINI_API_BASE_URL': (
        'Custom base URL of the Gemini Developer API, e.g. a reverse proxy of '
        '`https://generativelanguage.googleapis.com` (native Gemini API, not OpenAI-compatible; '
        'without `/v1beta`). Chat, model list, Files API and Interactions requests all go there. '
        'Ignored with Vertex AI.\n\n'
        '*Admin-only: users cannot override it, because the requests carry the API key.*'
    ),
    # Vertex AI
    'USE_VERTEX_AI': (
        'Whether to use Google Cloud Vertex AI instead of the standard Gemini API.\n\n'
        '*Requires `VERTEX_PROJECT` to be set.*'
    ),
    'VERTEX_PROJECT': 'Google Cloud project ID to use with Vertex AI.',
    'VERTEX_LOCATION': 'Google Cloud region/location for Vertex AI (e.g., `global`, `us-central1`).',
    # Routing & Fallback
    'ENABLE_FREE_TIER_FALLBACK': (
        'Automatically switch to the Paid API if a Free API request fails due to quota limits (`429`)'
        ' or model overload (`503`).\n\n'
        '*Requires both Free and Paid API keys to be configured.*'
    ),
    'TASK_MODEL_ROUTING': (
        'Determines API routing strategy for task models (e.g. title generation):\n'
        '- `only_free`: Use only the Free API.\n'
        '- `free_fallback`: Try Free API first, fallback to Paid on failure.\n'
        '- `only_paid`: Bypass Free API and use Paid API directly (or Vertex AI if enabled).\n'
        '- `match_main`: Follow the same routing logic as the main chat generation.'
    ),
    # Filtering & Cache
    'MODEL_WHITELIST': (
        'Comma-separated list of allowed model names.\n\n'
        'Supports `fnmatch` wildcard patterns: `*`, `?`, `[seq]`, `[!seq]`.'
    ),
    'MODEL_BLACKLIST': (
        'Comma-separated list of blacklisted model names.\n\n'
        'Supports `fnmatch` wildcard patterns: `*`, `?`, `[seq]`, `[!seq]`.'
    ),
    'CACHE_MODELS': (
        'Whether to cache available models on startup and refresh only when whitelist or blacklist rules change.'
    ),
    # Features & Execution Parameters
    'THINKING_CONFIG_RULES': (
        'Thinking parameter configuration. Can be an `int` (budget), a level `str`'
        ' (`MINIMAL`, `LOW`, `MEDIUM`, `HIGH`, `XHIGH`, or `NONE` for as little thinking as the model allows),'
        ' `None`, or a JSON string mapping model name regex patterns to budget (`int`) or level (`str`).\n\n'
        'The value is fitted to each model: Gemini 3 and 4 get their nearest supported `thinking_level`'
        ' (a budget is converted, since Gemini 4 deprecates `thinking_budget`), Gemini 2.5 gets a'
        ' `thinking_budget` within its range (a level is converted). The chat parameter `reasoning_effort`'
        ' overrides this valve.\n\n'
        'When using a JSON string mapping, **key order matters:** the first matching regex pattern is used,'
        ' so place more specific patterns before broader catch-all patterns.\n\n'
        'See [Gemini thinking](https://aistudio.google.com/docs/thinking#levels-budgets).'
    ),
    'SHOW_THINKING_SUMMARY': 'Whether to display the thinking process summary in responses (Gemini 2.5 and later).',
    'USE_FILES_API': (
        'Whether to use the Google Files API for uploading files (enables caching and performance benefits).\n\n'
        'If disabled, raw file bytes are sent directly in request payloads.'
    ),
    'PARSE_YOUTUBE_URLS': (
        'Whether to parse YouTube video URLs from user messages and provide content as context.\n\n'
        'If disabled, YouTube links are treated as plain text.'
    ),
    'USE_ENTERPRISE_SEARCH': (
        'Enable Enterprise Search tool allowing models to fetch and ground content from specified web URLs.'
    ),
    'MAPS_GROUNDING_COORDINATES': (
        'Latitude and longitude coordinates for location-aware Google Maps grounding.\n\n'
        'Expected format: `latitude,longitude` (e.g., `40.7128,-74.0060`).'
    ),
    'MEDIA_RESOLUTION': (
        'Resolution at which images, video frames and PDF pages are tokenized (more tokens, finer detail).\n\n'
        '`ULTRA_HIGH` applies to images only and needs Gemini 3 or later (per-part resolution);'
        ' other media then use `HIGH`. If set to `None`, the model default is used.'
    ),
    'FILE_SEARCH_STORE_NAMES': (
        'Comma-separated File Search store names (`fileSearchStores/...`). When set, models that support it'
        ' ground their answers in these stores.\n\n'
        'File Search cannot be combined with Google Search, URL context, Maps or code execution;'
        ' those take precedence when they are switched on for a message.'
    ),
    'FILE_SEARCH_METADATA_FILTER': (
        'Optional File Search metadata filter in AIP-160 syntax (e.g., `author="Jane Doe"`).'
    ),
    'OMNI_VIDEO_ASPECT_RATIO': 'Aspect ratio of Omni videos. If set to `None`, the model default (`16:9`) is used.',
    'OMNI_VIDEO_RESOLUTION': 'Resolution of Omni videos. If set to `None`, the model default (`720p`) is used.',
    'OMNI_VIDEO_DURATION': (
        'Length of Omni videos in seconds (3-15). If set to `None`, the model default (10 seconds) is used.'
    ),
    'IMAGE_SEARCH': (
        'Whether image models that support it (Nano Banana 2.1, Gemini 3.1 Flash Image) also use Google Image'
        ' Search when Google Search grounding is switched on.'
    ),
    'IMAGE_RESOLUTION': (
        'Output resolution for generated images (`512` is available on Gemini 3.1 Flash Image only).\n\n'
        "If set to `None`, Google's backend uses its default value (typically `1K`)."
        ' A value the model does not support is not sent.'
    ),
    'IMAGE_ASPECT_RATIO': (
        'Aspect ratio for image generation.\n\n'
        "If set to `None`, Google's backend uses its default value (matching aspect ratio for image editing,"
        ' or `1:1` otherwise). A value the model does not support is not sent.'
    ),
    # Interactions API models
    'ENABLE_DEEP_RESEARCH': (
        'Offer the Deep Research agents (Interactions API) as models. A research task runs in the background for'
        ' up to an hour and typically costs about $1-3 (Deep Research) or $3-7 (Deep Research Max).'
    ),
    'ENABLE_OMNI_VIDEO': (
        'Offer Gemini Omni 1.2 Flash video generation (Interactions API) as a model. Paid tier only; a 10 second'
        ' 720p video costs about $0.70.'
    ),
}


def _format_valve_desc(text: str, default: Any = None, is_user: bool = False) -> str:
    """Formats Markdown descriptions for Valves and UserValves fields."""
    text = text.strip()
    sep = '\n\n---\n\n'
    if is_user:
        return f"{text}\n\n*If not set, the admin's setting is used.*{sep}"
    formatted_default = f'`{default}`' if default is not None else '`None`'
    return f'{text}\n\n**Default:** {formatted_default}{sep}'


def _admin_field(name: str, default: Any = None, admin_section_start: bool = False) -> Any:
    """Helper to construct a Pydantic Field with default value and formatted admin description."""
    raw_desc = _VALVE_DESCRIPTIONS.get(name, '')
    desc = _format_valve_desc(raw_desc, default=default)
    if admin_section_start:
        desc = f'{desc}### Admin-Only Options'
    return Field(default=default, description=desc)


def _is_thinking_level(value: str) -> bool:
    """Whether a THINKING_CONFIG_RULES string names a thinking level (or NONE)."""
    return value.strip().upper() in (*_THINKING_LEVELS, _THINKING_OFF)


class _SharedValves(BaseModel):
    """Base model holding user-configurable options and shared field validators."""

    GEMINI_FREE_API_KEY: str | None = _admin_field('GEMINI_FREE_API_KEY')
    GEMINI_PAID_API_KEY: str | None = _admin_field('GEMINI_PAID_API_KEY')
    USE_VERTEX_AI: bool = _admin_field('USE_VERTEX_AI', False)
    VERTEX_PROJECT: str | None = _admin_field('VERTEX_PROJECT')
    VERTEX_LOCATION: str = _admin_field('VERTEX_LOCATION', 'global')
    ENABLE_FREE_TIER_FALLBACK: bool = _admin_field('ENABLE_FREE_TIER_FALLBACK', False)
    TASK_MODEL_ROUTING: Literal[
        'only_free',
        'free_fallback',
        'only_paid',
        'match_main',
    ] = _admin_field('TASK_MODEL_ROUTING', 'match_main')
    THINKING_CONFIG_RULES: StrictInt | StrictStr | None = _admin_field('THINKING_CONFIG_RULES', '{"$a": 0}')
    SHOW_THINKING_SUMMARY: bool = _admin_field('SHOW_THINKING_SUMMARY', True)
    USE_FILES_API: bool = _admin_field('USE_FILES_API', True)
    PARSE_YOUTUBE_URLS: bool = _admin_field('PARSE_YOUTUBE_URLS', True)
    MAPS_GROUNDING_COORDINATES: str | None = _admin_field('MAPS_GROUNDING_COORDINATES')
    MEDIA_RESOLUTION: Literal['LOW', 'MEDIUM', 'HIGH', 'ULTRA_HIGH'] | None = _admin_field('MEDIA_RESOLUTION')
    FILE_SEARCH_STORE_NAMES: str | None = _admin_field('FILE_SEARCH_STORE_NAMES')
    FILE_SEARCH_METADATA_FILTER: str | None = _admin_field('FILE_SEARCH_METADATA_FILTER')
    OMNI_VIDEO_ASPECT_RATIO: Literal['16:9', '9:16'] | None = _admin_field('OMNI_VIDEO_ASPECT_RATIO')
    OMNI_VIDEO_RESOLUTION: Literal['360p', '720p', '1080p', '4k'] | None = _admin_field('OMNI_VIDEO_RESOLUTION')
    OMNI_VIDEO_DURATION: StrictInt | None = _admin_field('OMNI_VIDEO_DURATION')
    IMAGE_SEARCH: bool = _admin_field('IMAGE_SEARCH', True)
    IMAGE_RESOLUTION: Literal['512', '1K', '2K', '4K'] | None = _admin_field('IMAGE_RESOLUTION')
    IMAGE_ASPECT_RATIO: (
        Literal[
            '1:1',
            '1:4',
            '1:8',
            '2:3',
            '3:2',
            '3:4',
            '4:1',
            '4:3',
            '4:5',
            '5:4',
            '8:1',
            '9:16',
            '16:9',
            '21:9',
        ]
        | None
    ) = _admin_field('IMAGE_ASPECT_RATIO', admin_section_start=True)

    @field_validator('THINKING_CONFIG_RULES', mode='after')
    @classmethod
    def validate_thinking_config_rules(cls, v: StrictInt | StrictStr | None) -> StrictInt | StrictStr | None:
        """Validates THINKING_CONFIG_RULES as an int (>= -1), level str, JSON str dict, or None."""
        if v is None:
            return v

        if isinstance(v, bool):
            raise ValueError('Thinking budget must be an integer, not a boolean.')

        if isinstance(v, int):
            if v < -1:
                raise ValueError('Thinking budget integer must be -1 or greater.')
            return v

        if isinstance(v, str):
            s = v.strip()
            if not s:
                return v

            if _is_thinking_level(s):
                return v

            try:
                data = json.loads(s)
            except json.JSONDecodeError:
                data = None

            if isinstance(data, dict):
                try:
                    for key, val in data.items():
                        re.compile(key)
                        if isinstance(val, bool):
                            raise ValueError(f"Value for pattern '{key}' must be an int or str, got bool.")
                        if isinstance(val, int):
                            if val < -1:
                                raise ValueError(f"Budget for pattern '{key}' must be -1 or greater.")
                        elif isinstance(val, str):
                            val_str = val.strip()
                            if not _is_thinking_level(val_str):
                                try:
                                    num = int(val_str)
                                except ValueError:
                                    raise ValueError(
                                        f"Value for pattern '{key}' must be an int (>= -1), "
                                        f'numeric string (>= -1), or level string '
                                        f'({"/".join((*_THINKING_LEVELS, _THINKING_OFF))}). Got: {val!r}'
                                    ) from None
                                if num < -1:
                                    raise ValueError(f"Budget for pattern '{key}' must be -1 or greater.")
                        else:
                            raise ValueError(
                                f"Value for pattern '{key}' must be an int or str, got {type(val).__name__}."
                            )
                except re.error as e:
                    raise ValueError(f'Invalid regex pattern in THINKING_CONFIG_RULES JSON dict: {e}') from e
                return v

            # If JSON parsing produced a non-dict or failed, treat as a numeric string.
            try:
                val_int = int(s)
            except ValueError:
                raise ValueError(
                    'THINKING_CONFIG_RULES must be an int (>= -1), level str '
                    f'({", ".join((*_THINKING_LEVELS, _THINKING_OFF))}), '
                    f'a valid JSON object mapping regex patterns to int/str, or None. Got: {v!r}'
                ) from None
            if val_int < -1:
                raise ValueError('Thinking budget integer must be -1 or greater.')
            return v

        raise ValueError(
            f'Invalid type for THINKING_CONFIG_RULES: {type(v).__name__}. Expected int (>= -1), str, or None.'
        )

    @field_validator('MAPS_GROUNDING_COORDINATES', mode='after')
    @classmethod
    def validate_coordinates_format(cls, v: str | None):
        """Reusable validator for 'latitude,longitude' format."""
        if v is not None and v != '':
            try:
                parts = v.split(',')
                if len(parts) != 2:
                    raise ValueError('Must contain exactly two parts separated by a comma.')

                lat_str, lon_str = parts
                lat = float(lat_str.strip())
                lon = float(lon_str.strip())

                if not (-90 <= lat <= 90):
                    raise ValueError('Latitude must be between -90 and 90.')
                if not (-180 <= lon <= 180):
                    raise ValueError('Longitude must be between -180 and 180.')
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid format for MAPS_GROUNDING_COORDINATES: '{v}'. "
                    f"Expected 'latitude,longitude' (e.g., '40.7128,-74.0060'). Original error: {e}"
                )
        return v

    @field_validator('IMAGE_RESOLUTION', mode='before')
    @classmethod
    def normalize_image_resolution(cls, v: Any) -> Any:
        """Accepts the earlier spellings of the 512 px size (`512PX`, `0.5K`)."""
        if isinstance(v, str) and v.strip().upper() in ('512PX', '0.5K'):
            return '512'
        return v

    @field_validator('OMNI_VIDEO_DURATION', mode='after')
    @classmethod
    def validate_video_duration(cls, v: int | None) -> int | None:
        """Omni accepts whole seconds from 3 to 15."""
        if v is not None and not 3 <= v <= 15:
            raise ValueError('OMNI_VIDEO_DURATION must be between 3 and 15 seconds.')
        return v


def _generate_user_valves(shared_cls: type[BaseModel]) -> type[BaseModel]:
    """Generates Pipe.UserValves model from _SharedValves with Optional fields and user descriptions."""
    fields: dict[str, Any] = {}
    for name, field_info in shared_cls.model_fields.items():
        raw_desc = _VALVE_DESCRIPTIONS.get(name, '')
        user_desc = _format_valve_desc(raw_desc, is_user=True)

        ann = field_info.annotation
        if get_origin(ann) is Literal:
            args = get_args(ann)
            user_ann = (
                Literal[(*args, '')] | None if '' not in args else ann | None  # type: ignore[operator]
            )
        elif ann is not None:
            user_ann = ann | None
        else:
            user_ann = Any

        fields[name] = (user_ann, Field(default=None, description=user_desc))

    return create_model('UserValves', __base__=shared_cls, **fields)  # type: ignore[call-overload]


# endregion Valves


class Pipe:
    class Valves(_SharedValves):
        # Admin-only: requests to this URL carry the admin's API key unless the user brings their own.
        GEMINI_API_BASE_URL: str | None = _admin_field('GEMINI_API_BASE_URL')
        USER_MUST_PROVIDE_AUTH_CONFIG: bool = _admin_field('USER_MUST_PROVIDE_AUTH_CONFIG', False)
        AUTH_WHITELIST: str | None = _admin_field('AUTH_WHITELIST')
        MODEL_WHITELIST: str = _admin_field('MODEL_WHITELIST', '*')
        MODEL_BLACKLIST: str | None = _admin_field('MODEL_BLACKLIST')
        CACHE_MODELS: bool = _admin_field('CACHE_MODELS', True)
        USE_ENTERPRISE_SEARCH: bool = _admin_field('USE_ENTERPRISE_SEARCH', False)
        ENABLE_DEEP_RESEARCH: bool = _admin_field('ENABLE_DEEP_RESEARCH', False)
        ENABLE_OMNI_VIDEO: bool = _admin_field('ENABLE_OMNI_VIDEO', False)

    UserValves = _generate_user_valves(_SharedValves)

    def __init__(self):
        self.valves = self.Valves()
        self.file_content_cache = SimpleMemoryCache(serializer=NullSerializer())
        self.file_id_to_hash_cache = SimpleMemoryCache(serializer=NullSerializer())
        # The embedded model table. The companion filter reads `gemini_model_config` from the loaded pipe for
        # its capability checks, so both always agree without hosting the YAML anywhere.
        self.catalog = ModelCatalog.embedded()
        self.gemini_model_config = self.catalog.table
        # State of each response across the rounds of Open WebUI's tool loop (see `_turn_state`).
        self._turns: dict[str, _TurnState] = {}
        log.success('Function has been initialized.')

    async def pipes(self) -> list[ModelData]:
        """Register all available Google models."""
        log.debug('pipes method has been called.')

        # Clear cache if caching is disabled
        if not self.valves.CACHE_MODELS:
            log.debug('CACHE_MODELS is False, clearing model cache.')
            cache_instance = getattr(self._get_genai_models, 'cache')
            await cast(BaseCache, cache_instance).clear()

        log.info('Fetching and filtering models from Google API.')
        # Get and filter models (potentially cached based on API key, base URL, white- and blacklist)
        try:
            client_args = self._prepare_client_args(self.valves)
            client_args += [self.valves.MODEL_WHITELIST, self.valves.MODEL_BLACKLIST]
            filtered_models = await self._get_genai_models(*client_args)
        except GenaiApiError:
            error_msg = 'Error getting the models from Google API, check the logs.'
            return [self._return_error_model(error_msg, exception=True)]

        filtered_models = [*filtered_models, *self._interactions_models()]
        log.info(f'Returning {len(filtered_models)} models to Open WebUI.')
        log.debug('Model list:', payload=[model['id'] for model in filtered_models])
        log.trace(
            'List of dicts that Pipe.pipes() will return:',
            payload=filtered_models,
            _log_truncation_enabled=False,
        )
        log.debug('pipes method has finished.')

        return filtered_models

    async def pipe(
        self,
        body: Body,
        __user__: UserData,
        __request__: Request,
        __metadata__: Metadata,
        __event_emitter__: Any = None,
    ) -> AsyncGenerator[dict | str, None] | dict:

        # FIXME: emit closing status message if error happens
        log.debug(f'pipe method has been called. Gemini Manifold Plus version is {VERSION}')
        log.trace('__metadata__:', payload=__metadata__)
        features = __metadata__.get('features', {}) or cast('Features', {})

        self._check_companion_filter_version(features)

        app_state: State = __request__.app.state
        model_id = self._get_model_name(body)
        __metadata__['canonical_model_id'] = model_id

        options = _resolve_options(
            self.valves,
            __user__.get('valves'),
            __user__.get('email'),
            body,
            __metadata__,
        )

        if task_type := __metadata__.get('task'):
            log.info(f'{task_type=}, disabling event emissions, YouTube URL parsing and document processing.')
            # We disable YouTube parsing for task models to minimize latency and token costs,
            # as simple tasks like title or tag generation do not require video context.
            options.PARSE_YOUTUBE_URLS = False

        execution_order = await self._determine_execution_order(
            options,
            __metadata__,
            features,
        )

        chat_id = __metadata__.get('chat_id')
        message_id = __metadata__.get('message_id')
        log.debug(f'Chat ID: {chat_id}, Message ID: {message_id}')

        event_emitter = self._get_event_emitter(
            app_state=app_state,
            chat_id=chat_id,
            message_id=message_id,
            is_task=bool(__metadata__.get('task')),
        )

        for attempt_idx, tier in enumerate(execution_order):
            is_last_attempt = attempt_idx == len(execution_order) - 1

            # Create a "Tier-Specific" options object for this attempt.
            # We copy to ensure we don't pollute the original options or metadata.
            current_options = copy.copy(options)

            if tier == 'free':
                current_options.GEMINI_PAID_API_KEY = None
                current_options.USE_VERTEX_AI = False
                __metadata__['is_paid_api'] = False
            elif tier == 'paid':
                current_options.GEMINI_FREE_API_KEY = None
                current_options.USE_VERTEX_AI = False
                __metadata__['is_paid_api'] = True
            elif tier == 'vertex':
                current_options.USE_VERTEX_AI = True
                __metadata__['is_paid_api'] = True

            try:
                log.info(f'Starting generation attempt on tier: {tier}')

                # Execute the attempt. This encapsulates client creation,
                # file uploads (scoped to key), and the API call.
                return await self._execute_generation_attempt(
                    tier=tier,
                    options=current_options,
                    body=body,
                    __user__=__user__,
                    __metadata__=__metadata__,
                    __request__=__request__,
                    event_emitter=event_emitter,
                    __event_emitter__=__event_emitter__,
                )

            except Exception as e:
                error_str = str(e).upper()

                # We catch Quota (429), Permission (403), and Overload/Unavailable (503) errors.
                # 503 is crucial for Free Tier which has strict load balancing.
                is_fallback_eligible = (
                    '429' in error_str
                    or '403' in error_str
                    or '503' in error_str
                    or 'UNAVAILABLE' in error_str
                    or (isinstance(e, genai_errors.ClientError) and e.code in [429, 403])
                    or (isinstance(e, genai_errors.ServerError) and e.code in [503])
                )

                should_retry = not is_last_attempt and tier == 'free' and is_fallback_eligible

                if should_retry:
                    reason = 'quota exceeded' if '429' in error_str else 'model overloaded'
                    log.warning(f'Free Tier {reason} (Error: {e}). Switching to Paid API...')
                    event_emitter.emit_status(f'Free Tier {reason}, switching to Paid API...', done=False)
                    continue

                # If we can't retry, re-raise the error to stop execution
                log.error(f'Error during request execution (Tier: {tier}): {e}')
                raise e

        raise ValueError('Exhausted execution options without result.')

    # region 1. Helper methods inside the Pipe class

    # region 1.1 Client initialization
    @staticmethod
    @cache
    def _get_or_create_genai_client(
        free_api_key: str | None = None,
        paid_api_key: str | None = None,
        base_url: str | None = None,
        use_vertex_ai: bool | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
    ) -> genai.Client:
        """
        Creates a genai.Client instance or retrieves it from cache.
        Raises GenaiApiError on failure.
        """

        # Prioritize the free key, then fall back to the paid key.
        api_key = free_api_key or paid_api_key

        if not vertex_project and not api_key:
            # FIXME: More detailed reason in the exception (tell user to set the API key).
            msg = 'Neither VERTEX_PROJECT nor a Gemini API key (free or paid) is set.'
            raise GenaiApiError(msg)

        if use_vertex_ai and vertex_project:
            kwargs = {
                'vertexai': True,
                'project': vertex_project,
                'location': vertex_location,
            }
            api = 'Vertex AI'
        else:  # Covers (use_vertex_ai and not vertex_project) OR (not use_vertex_ai)
            if use_vertex_ai and not vertex_project:
                log.warning('Vertex AI is enabled but no project is set. Using Gemini Developer API.')
            # This also implicitly covers the case where api_key might be None,
            # which is handled by the initial check or the SDK.
            kwargs = {
                'api_key': api_key,
                # google-genai>=3.0.0 defaults the Developer API to `v1`. Pin `v1beta` so the pipe keeps the
                # endpoints it was built against; e.g. the `gemini-*-latest` model aliases only exist on `v1beta`.
                'http_options': types.HttpOptions(base_url=base_url, api_version='v1beta'),
            }
            api = 'Gemini Developer API'

        try:
            client = genai.Client(**kwargs)
            log.success(f'{api} Genai client successfully initialized.')
            return client
        except Exception as e:
            raise GenaiApiError(f'{api} Genai client initialization failed: {e}') from e

    def _get_user_client(self, options: ResolvedOptions, user_email: str) -> genai.Client:
        user_whitelist = options.AUTH_WHITELIST.split(',') if options.AUTH_WHITELIST else []
        log.debug(
            f'User whitelist: {user_whitelist}, user email: {user_email}, '
            f'USER_MUST_PROVIDE_AUTH_CONFIG: {options.USER_MUST_PROVIDE_AUTH_CONFIG}'
        )
        if options.USER_MUST_PROVIDE_AUTH_CONFIG and user_email not in user_whitelist:
            if not options.GEMINI_FREE_API_KEY and not options.GEMINI_PAID_API_KEY:
                error_msg = (
                    'User must provide their own authentication configuration. '
                    'Please set GEMINI_FREE_API_KEY or GEMINI_PAID_API_KEY in UserValves,'
                    ' model parameters or chat parameters.'
                )
                raise ValueError(error_msg)
        try:
            client_args = self._prepare_client_args(options)
            client = self._get_or_create_genai_client(*client_args)
        except GenaiApiError as e:
            error_msg = f'Failed to initialize genai client for user {user_email}: {e}'
            # FIXME: include correct traceback.
            raise ValueError(error_msg) from e
        return client

    @staticmethod
    def _prepare_client_args(
        source_valves: Pipe.Valves | _SharedValves,
    ) -> list[str | bool | None]:
        """Prepares arguments for _get_or_create_genai_client from source_valves."""
        ATTRS = [
            'GEMINI_FREE_API_KEY',
            'GEMINI_PAID_API_KEY',
            'GEMINI_API_BASE_URL',
            'USE_VERTEX_AI',
            'VERTEX_PROJECT',
            'VERTEX_LOCATION',
        ]
        return [getattr(source_valves, attr, None) for attr in ATTRS]

    # endregion 1.1 Client initialization

    # region 1.2 Model retrival from Google API

    @cached()  # aiocache.cached for async method
    async def _get_genai_models(
        self,
        free_api_key: str | None = None,
        paid_api_key: str | None = None,
        base_url: str | None = None,
        use_vertex_ai: bool | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        whitelist_str: str = '*',
        blacklist_str: str | None = None,
    ) -> list[ModelData]:
        """
        Gets valid Google models from API(s) and filters them.
        If use_vertex_ai, vertex_project, and api_key are all provided,
        models are fetched from both Vertex AI and Gemini Developer API and merged.
        """
        all_raw_models: list[types.Model] = []

        # Condition for fetching from both sources
        fetch_both = bool(use_vertex_ai and vertex_project and (free_api_key or paid_api_key))

        if fetch_both:
            log.info('Attempting to fetch models from both Gemini Developer API and Vertex AI.')
            gemini_models_list: list[types.Model] = []
            vertex_models_list: list[types.Model] = []

            # TODO: perf, consider parallelizing these two fetches
            # 1. Fetch from Gemini Developer API
            try:
                gemini_client = self._get_or_create_genai_client(
                    free_api_key=free_api_key,
                    paid_api_key=paid_api_key,
                    base_url=base_url,
                    use_vertex_ai=False,  # Explicitly target Gemini API
                    vertex_project=None,
                    vertex_location=None,
                )
                gemini_models_list = await self._fetch_models_from_client_internal(
                    gemini_client, 'Gemini Developer API'
                )
            except GenaiApiError as e:
                log.warning(f'Failed to initialize or retrieve models from Gemini Developer API: {e}')
            except Exception as e:
                log.warning(
                    f'An unexpected error occurred with Gemini Developer API models: {e}',
                    exc_info=True,
                )

            # 2. Fetch from Vertex AI
            try:
                vertex_client = self._get_or_create_genai_client(
                    use_vertex_ai=True,  # Explicitly target Vertex AI
                    vertex_project=vertex_project,
                    vertex_location=vertex_location,
                    base_url=base_url,  # Pass base_url for potential Vertex custom endpoints
                )
                vertex_models_list = await self._fetch_models_from_client_internal(vertex_client, 'Vertex AI')
            except GenaiApiError as e:
                log.warning(f'Failemodel_configd to initialize or retrieve models from Vertex AI: {e}')
            except Exception as e:
                log.warning(
                    f'An unexpected error occurred with Vertex AI models: {e}',
                    exc_info=True,
                )

            # 3. Combine and de-duplicate
            # Prioritize models from Gemini Developer API in case of ID collision
            combined_models_dict: dict[str, types.Model] = {}

            for model in gemini_models_list:
                if model.name:
                    model_id = self._strip_api_prefix(model.name)
                    if model_id and model_id not in combined_models_dict:
                        combined_models_dict[model_id] = model
                else:
                    log.trace(f'Gemini model without a name encountered: {model.display_name or "N/A"}')

            for model in vertex_models_list:
                if model.name:
                    model_id = self._strip_api_prefix(model.name)
                    if model_id:
                        if model_id not in combined_models_dict:
                            combined_models_dict[model_id] = model
                        else:
                            log.info(
                                f"Duplicate model ID '{model_id}' from Vertex AI already sourced from"
                                ' Gemini API. Keeping Gemini API version.'
                            )
                else:
                    log.trace(f'Vertex AI model without a name encountered: {model.display_name or "N/A"}')

            all_raw_models = list(combined_models_dict.values())

            log.info(
                f'Fetched {len(gemini_models_list)} models from Gemini API, '
                f'{len(vertex_models_list)} from Vertex AI. '
                f'Combined to {len(all_raw_models)} unique models.'
            )

            if not all_raw_models and (gemini_models_list or vertex_models_list):
                log.warning(
                    'Models were fetched but resulted in an empty list after de-duplication,'
                    ' possibly due to missing names or empty/duplicate IDs.'
                )

            if not all_raw_models and not gemini_models_list and not vertex_models_list:
                raise GenaiApiError(
                    'Failed to retrieve models: Both Gemini Developer API and Vertex AI attempts yielded no models.'
                )

        else:  # Single source logic
            # Determine if we are effectively using Vertex AI or Gemini API
            # This depends on user's config (use_vertex_ai) and availability of project/key
            client_target_is_vertex = bool(use_vertex_ai and vertex_project)
            client_source_name = 'Vertex AI' if client_target_is_vertex else 'Gemini Developer API'
            log.info(f'Attempting to fetch models from a single source: {client_source_name}.')

            try:
                client = self._get_or_create_genai_client(
                    free_api_key=free_api_key,
                    paid_api_key=paid_api_key,
                    base_url=base_url,
                    use_vertex_ai=client_target_is_vertex,  # Pass the determined target
                    vertex_project=vertex_project if client_target_is_vertex else None,
                    vertex_location=(vertex_location if client_target_is_vertex else None),
                )
                all_raw_models = await self._fetch_models_from_client_internal(client, client_source_name)

                if not all_raw_models:
                    raise GenaiApiError(
                        f'No models retrieved from {client_source_name}. This could be due to an'
                        ' API error, network issue, or no models being available.'
                    )

            except GenaiApiError as e:
                raise GenaiApiError(f'Failed to get models from {client_source_name}: {e}') from e
            except Exception as e:
                log.error(
                    f'An unexpected error occurred while configuring client or fetching models'
                    f' from {client_source_name}: {e}',
                    exc_info=True,
                )
                raise GenaiApiError(
                    f'An unexpected error occurred while retrieving models from {client_source_name}: {e}'
                ) from e

        # --- Common processing for all_raw_models ---

        if not all_raw_models:
            log.warning('No models available after attempting all configured sources.')
            return []

        log.info(f'Processing {len(all_raw_models)} unique raw models.')

        generative_models: list[types.Model] = []
        for model in all_raw_models:
            if model.name is None:
                log.trace(f'Skipping model with no name during generative filter: {model.display_name or "N/A"}')
                continue
            actions = model.supported_actions
            model_id = self._strip_api_prefix(model.name)
            if any(fnmatch.fnmatch(model_id, pattern) for pattern in NON_CHAT_MODEL_PATTERNS):
                log.trace(f"Model '{model_id}' skipped, not a chat model this pipe can serve.")
            elif actions is None or 'generateContent' in actions:  # Includes models if actions is None (e.g., Vertex)
                generative_models.append(model)
            else:
                log.trace(f"Model '{model.name}' (ID: {model_id}) skipped, not generative (actions: {actions}).")

        if not generative_models:
            log.warning('No generative models found after filtering all retrieved models.')
            return []

        def match_patterns(name_to_check: str, list_of_patterns_str: str | None) -> bool:
            if not list_of_patterns_str:
                return False
            patterns = [
                pat for pat in list_of_patterns_str.replace(' ', '').split(',') if pat
            ]  # Ensure pat is not empty
            return any(fnmatch.fnmatch(name_to_check, pat) for pat in patterns)

        filtered_models_data: list[ModelData] = []
        for model in generative_models:
            # model.name is guaranteed non-None by generative_models filter logic
            assert model.name is not None
            stripped_name = self._strip_api_prefix(model.name)

            if not stripped_name:
                log.warning(
                    f"Model '{model.name}' (display: {model.display_name}) resulted in an empty ID"
                    ' after stripping. Skipping.'
                )
                continue

            passes_whitelist = not whitelist_str or match_patterns(stripped_name, whitelist_str)
            passes_blacklist = not blacklist_str or not match_patterns(stripped_name, blacklist_str)

            if passes_whitelist and passes_blacklist:
                filtered_models_data.append(
                    {
                        'id': stripped_name,
                        'name': model.display_name or stripped_name,
                        'description': model.description,
                    }
                )
            else:
                log.trace(
                    f"Model ID '{stripped_name}' filtered out by whitelist/blacklist. "
                    f'Whitelist match: {passes_whitelist}, Blacklist pass: {passes_blacklist}'
                )

        log.info(
            f'Filtered {len(generative_models)} generative models down to {len(filtered_models_data)}'
            ' models based on white/blacklists.'
        )
        return filtered_models_data

    # TODO: Use cache for this method too?
    async def _fetch_models_from_client_internal(self, client: genai.Client, source_name: str) -> list[types.Model]:
        """Helper to fetch models from a given client and handle common exceptions."""
        try:
            google_models_pager = await client.aio.models.list(
                config={'query_base': True}  # Fetch base models by default
            )
            models = [model async for model in google_models_pager]
            log.info(f'Retrieved {len(models)} models from {source_name}.')
            log.trace(f'All models returned by {source_name}:', payload=models)  # Can be verbose
            return models
        except Exception as e:
            log.error(f'Retrieving models from {source_name} failed: {e}')
            # Return empty list; caller decides if this is fatal for the whole operation.
            return []

    @staticmethod
    def _return_error_model(error_msg: str, warning: bool = False, exception: bool = True) -> ModelData:
        """Returns a placeholder model for communicating error inside the pipes method to the front-end."""
        if warning:
            log.opt(depth=1, exception=False).warning(error_msg)
        else:
            log.opt(depth=1, exception=exception).error(error_msg)
        return {
            'id': 'error',
            'name': '[gemini_manifold] ' + error_msg,
            'description': error_msg,
        }

    @staticmethod
    def _strip_api_prefix(model_name: str) -> str:
        """
        Extract the model identifier by removing API resource prefixes.
        e.g., "models/gemini-1.5-flash-001" -> "gemini-1.5-flash-001"
        e.g., "publishers/google/models/gemini-1.5-pro" -> "gemini-1.5-pro"
        Does NOT handle the manifold pipe prefix (e.g. "gemini_manifold_google_genai.").
        """
        # Remove everything up to the last '/'
        return model_name.split('/')[-1]

    @staticmethod
    def _get_model_name(body: Body) -> str:
        """
        Extracts the canonical model name from the request body.

        Open WebUI resolves custom workspace models to their base model and sends the manifold model id
        (`<function id>.<model id>`) in `body['model']`. Function ids are identifiers without dashes, while
        Gemini model ids always contain one, which tells a prefixed id from a bare one.

        Args:
            body: The request body dictionary.

        Returns:
            The canonical model name (prefix removed).
        """
        effective_model_name: str = body.get('model', '')
        initial_model_name = effective_model_name

        # A custom model's base model id, for callers that still send the request metadata in the body.
        if metadata := body.get('metadata'):
            if base_model_name := metadata.get('model', {}).get('info', {}).get('base_model_id'):
                effective_model_name = base_model_name

        prefix, dot, rest = effective_model_name.partition('.')
        canonical_model_name = rest if dot and '-' not in prefix else effective_model_name

        log.debug(
            f"Model Name Extraction: initial='{initial_model_name}', "
            f"effective='{effective_model_name}', canonical='{canonical_model_name}'"
        )
        return canonical_model_name

    @staticmethod
    def _parse_gemini_version(model_id: str) -> float:
        """Extracts the Gemini model version number from model ID.

        If not a Gemini model or version is missing, logs a warning and returns 3.0.
        """
        model_id_lower = model_id.lower()
        if 'gemini' in model_id_lower:
            match = re.search(r'gemini-(\d+(?:\.\d+)?)', model_id_lower)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

        log.warning(
            f"Model '{model_id}' is either not a Gemini model or missing a standard version number. "
            'Treating as Gemini 3+ model.'
        )
        return 3.0

    # endregion 1.2 Model retrival from Google API

    # region 1.3 GenerateContentConfig assembly

    @classmethod
    def _resolve_thinking_config(cls, rules_val: int | str | None, model_id: str) -> tuple[int | None, str | None]:
        """Evaluates THINKING_CONFIG_RULES (int, level str, or JSON regex dict) against model_id
        to determine the requested budget/level. `_fit_thinking` maps the request onto the model."""
        if rules_val is None:
            return None, None

        if isinstance(rules_val, int):
            return rules_val, None

        if isinstance(rules_val, str):
            s = rules_val.strip()
            if not s:
                return None, None

            if _is_thinking_level(s):
                return None, s.upper()

            try:
                budget = int(s)
                return budget, None
            except ValueError:
                pass

            try:
                rules = json.loads(s)
                if not isinstance(rules, dict):
                    return None, None
            except Exception as e:
                log.warning(f'Failed to parse THINKING_CONFIG_RULES JSON: {e}')
                return None, None

            level_hits: list[str] = []
            budget_hits: list[int] = []
            for pattern, val in rules.items():
                try:
                    if not re.search(pattern, model_id):
                        continue
                except re.error as e:
                    log.warning(f"Invalid regex pattern '{pattern}' in THINKING_CONFIG_RULES: {e}")
                    continue
                if isinstance(val, int) and not isinstance(val, bool):
                    budget_hits.append(val)
                elif isinstance(val, str):
                    v_str = val.strip()
                    if _is_thinking_level(v_str):
                        level_hits.append(v_str.upper())
                    else:
                        try:
                            budget_hits.append(int(v_str))
                        except ValueError:
                            pass

            # Budget-only models (Gemini 2.5) prefer a budget rule, level-based models a level rule.
            if cls._parse_gemini_version(model_id) < 3.0:
                if budget_hits:
                    return budget_hits[0], None
                if level_hits:
                    return None, level_hits[0]
            else:
                if level_hits:
                    return None, level_hits[0]
                if budget_hits:
                    return budget_hits[0], None

        return None, None

    def _fit_thinking(self, model_id: str, budget: int | None, level: str | None) -> dict[str, Any]:
        """
        Maps a requested budget or level onto what the model accepts (`thinking` in the model table).

        Models with thinking levels (Gemini 3 and 4) get the nearest supported `thinking_level` and never a
        `thinking_budget` (deprecated on Gemini 4, and rejected together with a level). Budget models
        (Gemini 2.5) get a `thinking_budget` within their range, levels mapped like the OpenAI
        compatibility layer does. NONE asks for as little thinking as the model allows.
        """
        spec = self.catalog.thinking(model_id)
        levels = [str(level_name).upper() for level_name in spec.get('levels') or []]
        if levels:
            if level is None and budget in (None, -1):
                return {}
            if level is None:
                level = _THINKING_OFF if budget == 0 else _budget_to_level(cast(int, budget))
            target = levels[0] if level == _THINKING_OFF else _nearest_level(level, levels)
            return {'thinking_level': types.ThinkingLevel[target]}

        budget_range = spec.get('budget')
        if isinstance(budget_range, dict):
            low, high = int(budget_range.get('min', 0)), int(budget_range.get('max', 24576))
            if level is not None:
                budget = 0 if level == _THINKING_OFF else _LEVEL_TO_BUDGET[level]
            if budget is None or budget == -1:
                return {} if budget is None else {'thinking_budget': -1}
            if budget == 0:
                return {'thinking_budget': 0 if spec.get('can_disable') else low}
            return {'thinking_budget': min(max(budget, low), high)}

        # Thinking without controls (e.g. Gemini 3 Pro Image): only thought summaries can be requested.
        return {}

    async def _thinking_config(
        self,
        options: ResolvedOptions,
        model_id: str,
        __metadata__: Metadata,
    ) -> types.ThinkingConfig | None:
        """The thinking config of a request, or None for models without thinking (per the model table)."""
        if not self.catalog.capability(model_id, 'thinking'):
            log.debug(f"Model '{model_id}' is not a thinking model in the model table.")
            return None

        # Resolved THINKING_CONFIG_RULES from options hierarchy
        # (Valves -> User Valves -> Model Params -> Chat Params, e.g. `reasoning_effort`)
        budget, level = self._resolve_thinking_config(options.THINKING_CONFIG_RULES, model_id)

        # Toggle filter override (gemini_reasoning_toggle in UI): OFF asks for as little thinking as possible.
        is_avail, is_on = await self._get_toggleable_feature_status('gemini_reasoning_toggle', __metadata__)
        if is_avail and not is_on:
            log.info(f"Reasoning toggle is OFF in UI; asking '{model_id}' for as little thinking as it allows.")
            budget, level = None, _THINKING_OFF

        thinking_kwargs = self._fit_thinking(model_id, budget, level)
        log.info(
            f"Thinking config for model '{model_id}': requested budget={budget}, level={level} -> {thinking_kwargs}"
        )
        return types.ThinkingConfig(include_thoughts=options.SHOW_THINKING_SUMMARY, **thinking_kwargs)

    async def _build_gen_content_config(
        self,
        body: Body,
        __metadata__: Metadata,
        options: ResolvedOptions,
        event_emitter: EventEmitter,
        tool_bridge: ToolBridge,
    ) -> types.GenerateContentConfig:
        """Assembles the GenerateContentConfig for a Gemini API request using ResolvedOptions."""
        features = __metadata__.get('features', {}) or {}
        is_vertex_ai = __metadata__.get('is_vertex_ai', False)
        # We are ensured to have a valid model ID at this point.
        model_id: str = __metadata__.get('canonical_model_id', '')

        log.debug(
            'Features extracted from metadata (UI toggles and config):',
            payload=features,
        )

        # Sampling parameters are deprecated (ignored by Gemini 3.6+, rejected by future models) and
        # `candidate_count` (`n`) is unsupported on Gemini 3+, so none of them is ever sent.
        if ignored := [key for key in ('temperature', 'top_p', 'top_k', 'n') if body.get(key) is not None]:
            log.debug(f'Not sending deprecated sampling parameters: {ignored}')

        # system_instruction is intentionally left unset here. It will be set by the caller.
        gen_content_conf = types.GenerateContentConfig(
            max_output_tokens=body.get('max_tokens') or body.get('max_completion_tokens'),
            stop_sequences=body.get('stop') or None,
            safety_settings=__metadata__.get('safety_settings'),
            thinking_config=await self._thinking_config(options, model_id, __metadata__),
            media_resolution=_global_media_resolution(options.MEDIA_RESOLUTION),
            # Open WebUI (or the API client) runs the tools; the SDK must never run them itself.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        gen_content_conf.response_modalities = ['TEXT']
        self._apply_response_format(gen_content_conf, body.get('response_format'), model_id, tool_bridge)

        # Optimization: Task models (titles, tags, search queries) do not require tools.
        # Disabling them here prevents unnecessary tool-use overhead and reduces latency.
        if __metadata__.get('task'):
            log.debug('Task model detected. Skipping tool configuration.')
            if tool_bridge.declarations:
                # Context compaction replays the chat's tools so that the request prefix matches the cached chat
                # request; the model must not call them.
                gen_content_conf.tools = [types.Tool(function_declarations=tool_bridge.declarations)]
                gen_content_conf.tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.NONE)
                )
            return gen_content_conf

        if self.catalog.is_image_model(model_id):
            gen_content_conf.response_modalities.append('IMAGE')
            gen_content_conf.image_config = self._image_config(model_id, options, event_emitter)

        builtin_tools, tool_config = await self._builtin_tools(features, options, __metadata__, event_emitter)
        declarations = tool_bridge.declarations
        if declarations and builtin_tools and not self._supports_tool_combination(model_id, is_vertex_ai):
            msg = (
                f"'{model_id}' cannot combine Google's built-in tools with function calling here, so the function"
                ' tools of this request were not offered to the model.'
            )
            if tool_bridge.user_selected:
                _log_and_toast(event_emitter, msg, 'info')
            else:
                log.info(msg)
            declarations = []

        tools = list(builtin_tools)
        if declarations:
            tools.append(types.Tool(function_declarations=declarations))
            if function_calling_config := tool_bridge.function_calling_config(body.get('tool_choice')):
                tool_config.function_calling_config = function_calling_config
            if builtin_tools:
                # Tool combination: the built-in tool calls come back as parts that must be sent back.
                tool_config.include_server_side_tool_invocations = True
        gen_content_conf.tools = tools or None
        if tool_config.model_dump(exclude_none=True):
            gen_content_conf.tool_config = tool_config
        return gen_content_conf

    async def _builtin_tools(
        self,
        features: Features,
        options: ResolvedOptions,
        __metadata__: Metadata,
        event_emitter: EventEmitter,
    ) -> tuple[list[types.Tool], types.ToolConfig]:
        """Google's built-in tools switched on for this request, and the tool config they need."""
        is_vertex_ai = __metadata__.get('is_vertex_ai', False)
        model_id: str = __metadata__.get('canonical_model_id', '')
        tools: list[types.Tool] = []
        tool_config = types.ToolConfig()

        if features.get('google_search_tool') and self.catalog.capability(model_id, 'search_grounding', True):
            tools.append(self._search_tool(options, model_id, is_vertex_ai))

        # NB: It is not possible to use both Search and Code execution at the same time,
        # however, it can be changed later, so let's just handle it as a common error
        if features.get('google_code_execution') and self.catalog.capability(model_id, 'code_execution', True):
            log.info('Using code execution on Google side.')
            tools.append(types.Tool(code_execution=types.ToolCodeExecution()))

        # Determine if URL context tool should be enabled. The toggle filter is the only switch.
        is_avail, is_on = await self._get_toggleable_feature_status('gemini_url_context_toggle', __metadata__)
        if is_avail and is_on:
            if not self.catalog.capability(model_id, 'url_context'):
                log.warning(
                    f'URL context tool is enabled, but model {model_id} does not support it'
                    ' (see capabilities.url_context in gemini_models.yaml). Skipping.'
                )
            elif is_vertex_ai and tools:
                log.warning('URL context tool is enabled, but Vertex AI is used with other tools. Skipping.')
            else:
                log.info(f'Model {model_id} is compatible with URL context tool. Enabling.')
                tools.append(types.Tool(url_context=types.UrlContext()))

        # Determine if Google Maps grounding should be enabled.
        is_avail, is_on = await self._get_toggleable_feature_status('gemini_maps_grounding_toggle', __metadata__)
        if is_avail and is_on and self.catalog.capability(model_id, 'grounding_google_maps', True):
            log.info('Enabling Google Maps grounding tool.')
            tools.append(types.Tool(google_maps=types.GoogleMaps()))

            if options.MAPS_GROUNDING_COORDINATES:
                # Validated by the valve's field validator.
                lat_str, lon_str = options.MAPS_GROUNDING_COORDINATES.split(',')
                lat_lng = types.LatLng(latitude=float(lat_str.strip()), longitude=float(lon_str.strip()))
                log.info(f'Using coordinates for Maps grounding: {lat_lng}')
                tool_config.retrieval_config = types.RetrievalConfig(lat_lng=lat_lng)

        stores = [name.strip() for name in (options.FILE_SEARCH_STORE_NAMES or '').split(',') if name.strip()]
        # File Search is a Gemini Developer API tool (Vertex AI has its own RAG Engine and Vertex AI Search).
        if stores and not is_vertex_ai and self.catalog.capability(model_id, 'file_search', True):
            if tools:
                _log_and_toast(
                    event_emitter,
                    'File Search cannot be combined with the Google tools switched on for this message,'
                    ' so it was not used.',
                    'info',
                )
            else:
                log.info(f'Using File Search over {len(stores)} store(s).')
                tools.append(
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=stores,
                            metadata_filter=options.FILE_SEARCH_METADATA_FILTER or None,
                        )
                    )
                )
        return tools, tool_config

    def _search_tool(self, options: ResolvedOptions, model_id: str, is_vertex_ai: bool) -> types.Tool:
        """Google Search grounding: Enterprise Web Search on Vertex AI if enabled, Image Search where supported."""
        if options.USE_ENTERPRISE_SEARCH and is_vertex_ai:
            log.info('Using grounding with Enterprise Web Search as a Tool.')
            return types.Tool(enterprise_web_search=types.EnterpriseWebSearch())
        if options.IMAGE_SEARCH and self.catalog.image_options(model_id).get('image_search') and not is_vertex_ai:
            log.info('Using grounding with Google Search (web and image search) as a Tool.')
            search_types = types.SearchTypes(web_search=types.WebSearch(), image_search=types.ImageSearch())
            return types.Tool(google_search=types.GoogleSearch(search_types=search_types))
        log.info('Using grounding with Google Search as a Tool.')
        return types.Tool(google_search=types.GoogleSearch())

    def _supports_tool_combination(self, model_id: str, is_vertex_ai: bool) -> bool:
        """Built-in tools together with function calling: Gemini 3 and later on the Gemini Developer API."""
        return not is_vertex_ai and self._parse_gemini_version(model_id) >= 3.0

    def _apply_response_format(
        self,
        gen_content_conf: types.GenerateContentConfig,
        response_format: Any,
        model_id: str,
        tool_bridge: ToolBridge,
    ) -> None:
        """Maps an OpenAI `response_format` (json_schema / json_object) to Gemini structured output."""
        if not isinstance(response_format, dict) or response_format.get('type') not in ('json_schema', 'json_object'):
            return
        if not self.catalog.capability(model_id, 'structured_outputs', True):
            log.warning(f"Model '{model_id}' does not support structured output; ignoring response_format.")
            return

        gen_content_conf.response_mime_type = 'application/json'
        json_schema = response_format.get('json_schema')
        if isinstance(json_schema, dict) and isinstance(json_schema.get('schema'), dict):
            gen_content_conf.response_json_schema = json_schema['schema']

        if tool_bridge.declarations and self._parse_gemini_version(model_id) < 3.0:
            # Structured output together with function calling needs Gemini 3 or later.
            log.warning(f"'{model_id}' cannot combine structured output with function calling; not offering tools.")
            tool_bridge.declarations = []

    def _image_config(
        self,
        model_id: str,
        options: ResolvedOptions,
        event_emitter: EventEmitter,
    ) -> types.ImageConfig | None:
        """The image config of an image model request; values the model does not support are not sent."""
        image_cfg = self.catalog.image_options(model_id)
        image_config_kwargs: dict[str, Any] = {}

        # Missing/empty resolutions means the model doesn't support `image_size`.
        if resolution := options.IMAGE_RESOLUTION:
            supported_resolutions = image_cfg.get('resolutions') or []
            if resolution in supported_resolutions:
                image_config_kwargs['image_size'] = resolution
            else:
                _log_and_toast(
                    event_emitter,
                    f"Image resolution '{resolution}' is not supported by '{model_id}'"
                    f' (supported: {", ".join(supported_resolutions) or "none"}); using the model default.',
                    'warning',
                )

        if aspect_ratio := options.IMAGE_ASPECT_RATIO:
            supported_aspect_ratios = image_cfg.get('aspect_ratios') or []
            if aspect_ratio in supported_aspect_ratios:
                image_config_kwargs['aspect_ratio'] = aspect_ratio
            else:
                _log_and_toast(
                    event_emitter,
                    f"Aspect ratio '{aspect_ratio}' is not supported by '{model_id}'"
                    f' (supported: {", ".join(supported_aspect_ratios)}); using the model default.',
                    'warning',
                )

        return types.ImageConfig(**image_config_kwargs) if image_config_kwargs else None

    # endregion 1.3 GenerateContentConfig assembly

    # region 1.4 Model response processing

    async def _aggregate_to_dict(
        self,
        generator: AsyncGenerator[dict | str, None],
        __event_emitter__: Any = None,
    ) -> dict:
        """
        Consumes the unified response generator and aggregates the chunks into a
        single OpenAI Chat Completion dictionary. This keeps our processing pipeline
        unified while properly satisfying OWUI's non-streaming request expectations.
        Events that a stream would carry (embeds, sources, files) go to the event emitter directly.
        """
        texts = {'content': '', 'reasoning_content': ''}
        lists: dict[str, list[dict]] = {'reasoning_details': [], 'tool_calls': []}
        usage = None
        error = None

        async for chunk in generator:
            if isinstance(chunk, str):
                # Skip string yields (like "data: [DONE]") used for streaming protocol
                continue

            if event := chunk.get('event'):
                if __event_emitter__:
                    await __event_emitter__(event)
                continue

            error = chunk.get('error') or error
            usage = chunk.get('usage') or usage
            delta = chunk['choices'][0].get('delta', {}) if chunk.get('choices') else {}
            for key in texts:
                texts[key] += delta.get(key) or ''
            for key in lists:
                lists[key].extend(delta.get(key) or [])

        tool_calls = [{k: v for k, v in call.items() if k != 'index'} for call in lists['tool_calls']]
        # Only add the optional keys if there was actually content for them
        message: dict[str, Any] = {
            'role': 'assistant',
            'content': texts['content'],
            **{
                key: value
                for key, value in (
                    ('reasoning_content', texts['reasoning_content']),
                    ('reasoning_details', lists['reasoning_details']),
                    ('tool_calls', tool_calls),
                )
                if value
            },
        }
        choice = {'index': 0, 'message': message, 'finish_reason': 'tool_calls' if tool_calls else 'stop'}
        return {'choices': [choice], 'usage': usage or {}, **({'error': error} if error else {})}

    @staticmethod
    def _request_config(
        config: types.GenerateContentConfig,
        contents: list[types.Content],
        options: ResolvedOptions,
        client: genai.Client,
    ) -> types.GenerateContentConfig:
        """The config sent to the SDK: a proxy may read only camelCase fields (see `_camel_case_request`)."""
        if options.GEMINI_API_BASE_URL and not client.vertexai:
            return _camel_case_request(config, contents)
        return config

    async def _execute_generation_attempt(
        self,
        tier: str,
        options: ResolvedOptions,
        body: Body,
        __user__: UserData,
        __metadata__: Metadata,
        __request__: Request,
        event_emitter: EventEmitter,
        __event_emitter__: Any = None,
    ) -> AsyncGenerator[dict | str, None] | dict:
        """
        Executes a single generation attempt with a specific tier configuration.
        Constructs a fresh client and file manager to ensure assets are
        scoped to the correct API key/project.
        """

        # 1. Client Creation
        client = self._get_user_client(options, __user__['email'])
        __metadata__['is_vertex_ai'] = client.vertexai
        api_name = 'Vertex AI Gemini API' if client.vertexai else 'Gemini Developer API'
        model_id = __metadata__.get('canonical_model_id', '')
        is_interactions_model = self._is_interactions_model(model_id, client)

        # 2. Files API Manager (Scoped to the current client)
        files_api_manager = FilesAPIManager(
            client=client,
            file_cache=self.file_content_cache,
            id_hash_cache=self.file_id_to_hash_cache,
            event_emitter=event_emitter,
        )

        # 3. Configuration Building (first: it decides which stored history parts can be sent back)
        tool_bridge = ToolBridge.for_request(body, __metadata__, accepts_functions=self._accepts_functions(model_id))
        gen_content_conf = (
            None
            if is_interactions_model
            else await self._build_gen_content_config(body, __metadata__, options, event_emitter, tool_bridge)
        )
        tool_config = gen_content_conf.tool_config if gen_content_conf else None

        # 4. Content Builder (Re-uploads files if client changed)
        builder = GeminiContentBuilder(
            messages_body=body.get('messages'),
            metadata_body=__metadata__,
            user_data=__user__,
            event_emitter=event_emitter,
            options=options,
            files_api_manager=files_api_manager,
            model_id=model_id,
            replay_server_side_parts=bool(tool_config and tool_config.include_server_side_tool_invocations),
            max_images=self.catalog.image_options(model_id).get('max_reference_images'),
            image_media_resolution=self._image_media_resolution(options, model_id),
        )

        event_emitter.emit_status('Preparing request...')
        contents = await builder.build_contents()

        if gen_content_conf is None:
            runner = InteractionsRunner(
                client=client,
                model_id=model_id,
                options=options,
                event_emitter=event_emitter,
                upload=lambda data, mime_type, name: self._upload_media(
                    data, mime_type, __metadata__, __request__.app, name
                ),
            )
            stream = runner.run(contents, _previous_interaction_id(body.get('messages') or []))
            return stream if body.get('stream', True) else await self._aggregate_to_dict(stream, __event_emitter__)

        gen_content_conf.system_instruction = builder.system_prompt

        # Check for image/system prompt compatibility
        is_image_model = self.catalog.is_image_model(model_id)
        if (is_image_model or 'gemma' in model_id) and gen_content_conf.system_instruction:
            gen_content_conf.system_instruction = None
            log.warning(f"Model '{model_id}' does not support system prompts. Removing.")

        gen_content_args = {
            'model': model_id,
            'contents': contents,
            'config': self._request_config(gen_content_conf, contents, options, client),
        }
        log.debug(f'Passing args to {api_name} (Tier: {tier}):', payload=gen_content_args)

        # 5. Stream Setup
        # 'is_streaming_request' tracks what Open WebUI expects to receive.
        # 'use_streaming_api' tracks how we actually call the Google SDK.
        is_streaming_request = body.get('stream', True)
        use_streaming_api = is_streaming_request

        # the Google GenAI SDK's streaming method often raises a "chunk too big" error
        # if a high-resolution image is requested. We avoid this by forcing
        # a non-streaming SDK call, while still yielding the result as a stream to OWUI.
        if use_streaming_api and options.IMAGE_RESOLUTION in ['2K', '4K']:
            log.info(
                f'Forcing non-streaming SDK call due to {options.IMAGE_RESOLUTION} resolution '
                "to avoid GenAI SDK 'chunk too big' errors."
            )
            use_streaming_api = False

        request_type_str = 'streaming' if use_streaming_api else 'non-streaming'
        request_type_msg = f'Sending {request_type_str} request to {api_name}...'
        log.info(request_type_msg)
        event_emitter.emit_status(request_type_msg)

        # 6. Execution & Peek Logic
        api_request_start_time = time.monotonic()
        if use_streaming_api:
            stream = await client.aio.models.generate_content_stream(**gen_content_args)  # type: ignore
        else:
            # When we use the non-streaming SDK call, we wrap the single response
            # in an iterator so that our unified processor can still treat it like a stream.
            response = await client.aio.models.generate_content(**gen_content_args)

            async def one_shot_iter():
                yield response

            stream = one_shot_iter()

        iterator = stream.__aiter__()

        try:
            first_chunk = await iterator.__anext__()
            first_token_time = time.monotonic()
        except StopAsyncIteration:
            raise ValueError('API returned an empty response stream.')

        # Success: Reconstruct the stream including the peeked chunk
        async def reconstructed_stream():
            yield first_chunk
            async for chunk in iterator:
                yield chunk

        log.info(f'Request successful ({tier}). Passing stream to unified processor.')

        processor = self._unified_response_processor(
            reconstructed_stream(),
            __request__.app,
            event_emitter,
            __metadata__,
            tool_bridge=tool_bridge,
            api_request_start_time=api_request_start_time,
            first_token_time=first_token_time,
        )

        # If OWUI requested a stream, we return the AsyncGenerator.
        # If OWUI requested a single object, we aggregate the chunks back into a dict.
        if is_streaming_request:
            return processor
        else:
            return await self._aggregate_to_dict(processor, __event_emitter__)

    def _check_free_tier_eligibility(
        self,
        model_id: str,
        features: Features,
        maps_requested: bool,
    ) -> bool:
        """
        Determines if the request is eligible for the Free Tier based on model config
        and requested features (e.g., grounding).
        """
        # 1. Check if model is configured as having a free tier in YAML
        pricing = self.catalog.entry(model_id).get('pricing', {})
        if not pricing.get('free_tier', False):
            return False

        # 2. Check for feature exclusions (e.g. Google Search is often Paid only)
        excluded_features = pricing.get('excluded_features', [])

        # Check Search
        is_search_requested = features.get('google_search_tool')
        if is_search_requested and 'search_grounding' in excluded_features:
            log.info(f'Free Tier ineligible: Search requested but excluded for {model_id}.')
            return False

        # Check Maps
        if maps_requested and 'grounding_google_maps' in excluded_features:
            log.info(f'Free Tier ineligible: Maps requested but excluded for {model_id}.')
            return False

        return True

    async def _unified_response_processor(
        self,
        response_stream: AsyncIterator[types.GenerateContentResponse],
        app: FastAPI,
        event_emitter: EventEmitter,
        __metadata__: Metadata,
        *,
        tool_bridge: ToolBridge,
        api_request_start_time: float | None = None,
        first_token_time: float | None = None,
    ) -> AsyncGenerator[dict | str, None]:
        """
        Processes an async iterator of GenerateContentResponse objects (one round of Open WebUI's tool loop),
        yielding structured dictionary chunks for the Open WebUI frontend.

        Answer and thought parts are yielded as they arrive. When the round ends, it yields the round's
        attributions, the model content that later requests must send back (`reasoning_details`), the
        function calls for Open WebUI to run (`tool_calls`) and the usage; then it does the post-processing.
        This single method handles both streaming and non-streaming (via an adapter) responses.
        """
        final_response_chunk: types.GenerateContentResponse | None = None
        error_occurred = False
        total_substitutions = 0
        first_chunk_received = False
        chunk_counter = 0
        in_think = False
        last_title: str | None = None
        thoughts_shown = False
        content_shown = False
        response_parts: list[types.Part] = []
        function_call_parts: list[types.Part] = []
        grounding_metadata: types.GroundingMetadata | None = None

        try:
            async for chunk in response_stream:
                candidate = self._get_first_candidate(chunk.candidates)
                content = candidate.content if candidate else None
                log.trace(
                    f"Processing response chunk #{chunk_counter}, first candidate's content:",
                    payload=content,
                )
                chunk_counter += 1
                final_response_chunk = chunk  # Keep the latest chunk for metadata
                if candidate and candidate.grounding_metadata:
                    grounding_metadata = candidate.grounding_metadata

                if not first_chunk_received:
                    # This is the first (and possibly only) chunk.
                    event_emitter.emit_status('Response received', done=True)
                    first_chunk_received = True

                if not (parts := chunk.parts):
                    log.warning('Chunk has no parts, skipping.')
                    continue

                response_parts.extend(parts)

                # This inner loop makes the method robust. It handles a single chunk
                # with many parts (non-streaming) or many chunks with one part (streaming).
                for part in parts:
                    if part.function_call:
                        # Sent as tool calls when the round ends; Open WebUI runs them and calls the pipe again.
                        function_call_parts.append(part)
                        continue

                    # Handle thought titles and transitions between reasoning and normal content.
                    if part.thought:
                        if not in_think:
                            # TODO: emit an status indicating that reasoning has started.
                            # include budget or level if set.
                            in_think = True

                        # Attempt to extract a title from any text within a thought part.
                        # TODO: refactor it to a helper method?
                        if isinstance(part.text, str):
                            try:
                                title: str | None = None
                                # Prefer markdown-style "### Heading" titles.
                                for m in re.finditer(r'(^|\n)###\s+(.+?)(?=\n|$)', part.text or ''):
                                    title = m.group(2).strip()
                                # Fall back to bold "**Title**" lines if no heading was found.
                                if not title:
                                    for m in re.finditer(
                                        r'(^|\n)\s*\*\*(.+?)\*\*\s*(?=\n|$)',
                                        part.text or '',
                                    ):
                                        title = (m.group(2) or '').strip()
                                if title:
                                    # Trim common surrounding quotes.
                                    title = title.strip('"“”‘’').strip()
                                if title and title != last_title:
                                    last_title = title
                                    event_emitter.emit_status(
                                        title,
                                        done=False,
                                        hidden=False,
                                        is_thought=True,
                                        indent_level=1,
                                    )
                            except Exception:
                                # Thought titles are a best-effort feature; failures should not break the stream.
                                pass
                    elif in_think:
                        # Terminate the 'in_think' state only when a non-thought part with actual content arrives.
                        # This prevents empty text parts from prematurely ending the thought block in the UI.
                        has_content = (
                            (isinstance(part.text, str) and part.text)
                            or part.inline_data
                            or part.executable_code
                            or part.code_execution_result
                        )
                        if has_content:
                            in_think = False
                            # Clear the last thought title when normal content begins.
                            event_emitter.emit_status(
                                'Thinking finished',
                                done=True,
                                is_thought=False,
                            )

                    payload, count = await self._process_part(
                        part,
                        app,
                        __metadata__,
                    )

                    if payload:
                        content_shown = content_shown or bool(payload.get('content'))
                        thoughts_shown = thoughts_shown or bool(payload.get('reasoning_content'))

                        if count > 0:
                            total_substitutions += count
                            log.debug(f'Disabled {count} special tag(s) in a part.')

                        structured_chunk = {'choices': [{'delta': payload}]}
                        yield structured_chunk

        except Exception as e:
            error_occurred = True
            error_msg = f'Response processing ended with error: {e}'
            log.exception(error_msg)
            # As a chunk, so the error reaches the chat with or without the companion's event emitter.
            yield {'error': {'detail': error_msg}}

        if total_substitutions > 0 and not error_occurred:
            plural_s = 's' if total_substitutions > 1 else ''
            toast_msg = (
                f'For clarity, {total_substitutions} special tag{plural_s} '
                'were disabled in the response by injecting a zero-width space (ZWS).'
            )
            event_emitter.emit_toast(toast_msg, 'info')

        if not error_occurred:
            for round_chunk in self._round_end_chunks(
                response_parts,
                function_call_parts,
                grounding_metadata,
                final_response_chunk,
                tool_bridge,
                __metadata__,
                thoughts_shown=thoughts_shown,
                content_shown=content_shown,
            ):
                yield round_chunk

        # Calculate usage data using the last received chunk.
        # If the chunk contains usage metadata, we yield it so the backend persists it to DB.
        if final_response_chunk and (
            usage_data := self._get_usage_data(
                final_response_chunk,
                __metadata__,
                event_emitter.start_time,
                api_request_start_time,
                first_token_time,
            )
        ):
            # Yielding this dictionary allows the OWUI proxy to catch and save usage.
            yield {'usage': usage_data}

        if not error_occurred:
            log.info('Response processing finished successfully!')

        try:
            await self._do_post_processing(
                final_response_chunk,
                event_emitter,
                app.state,
                __metadata__,
                stream_error_happened=error_occurred,
                final_round=not function_call_parts,
            )
        except Exception as e:
            error_msg = f'Post-processing failed with error:\n\n{e}'
            event_emitter.emit_toast(error_msg, 'error')
            log.exception(error_msg)

        log.debug('Unified response processor has finished.')

    def _round_end_chunks(
        self,
        response_parts: list[types.Part],
        function_call_parts: list[types.Part],
        grounding_metadata: types.GroundingMetadata | None,
        final_response_chunk: types.GenerateContentResponse | None,
        tool_bridge: ToolBridge,
        __metadata__: Metadata,
        *,
        thoughts_shown: bool,
        content_shown: bool,
    ) -> list[dict]:
        """
        The chunks that end a round, in the order Open WebUI needs them: the Google Maps attribution right
        after the text it supports, the stored model content (before the tool calls, so that Open WebUI
        keeps it with this round), the tool calls, and the Search Suggestions.
        """
        chunks: list[dict] = []
        if attribution := _maps_attribution(grounding_metadata):
            chunks.append({'choices': [{'delta': {'content': attribution}}]})

        stored_parts = _compact_parts(response_parts)
        if self._needs_replay(stored_parts, grounding_metadata, bool(function_call_parts), thoughts_shown):
            detail = {
                'type': 'reasoning.encrypted',
                'format': REPLAY_FORMAT,
                'index': 0,
                'data': types.Content(role='model', parts=stored_parts).model_dump_json(exclude_none=True),
            }
            chunks.append({'choices': [{'delta': {'reasoning_details': [detail]}}]})

        if function_call_parts:
            chunks.append({'choices': [{'delta': {'tool_calls': tool_bridge.tool_calls(function_call_parts)}}]})
            chunks.append({'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]})
        elif (
            not content_shown and (reason := self._finish_reason_name(final_response_chunk)) in FAILED_TOOL_CALL_REASONS
        ):
            # The model's tool call failed and it said nothing: tell the user instead of leaving the message empty.
            chunks.append({'choices': [{'delta': {'content': f'*{FINISH_REASON_DESCRIPTIONS[reason]} ({reason})*'}}]})

        if suggestions := self._search_suggestions(grounding_metadata, __metadata__):
            # The Search Suggestions must be shown with the grounded answer, unmodified: an embed renders
            # Google's HTML in its own frame. `replace` keeps the chat and the database in step.
            chunks.append({'event': {'type': 'embeds', 'data': {'embeds': suggestions, 'replace': True}}})
        return chunks

    @staticmethod
    def _needs_replay(
        stored_parts: list[types.Part],
        grounding_metadata: types.GroundingMetadata | None,
        has_function_calls: bool,
        thoughts_shown: bool,
    ) -> bool:
        """
        Whether a round's model content must be stored for later requests. Content that the visible text
        cannot reproduce (media, code execution, server-side tool parts, the text before citation markers)
        always is; thought signatures are when they are required (function calls) or when the round shows a
        reasoning block anyway. Optional text signatures alone are not worth an empty reasoning block.
        """
        structured = any(
            p.tool_call
            or p.tool_response
            or p.executable_code
            or p.code_execution_result
            or p.file_data
            or p.inline_data
            for p in stored_parts
        )
        signed = any(p.thought_signature for p in stored_parts)
        return structured or _is_grounded(grounding_metadata) or (signed and (has_function_calls or thoughts_shown))

    def _search_suggestions(
        self,
        grounding_metadata: types.GroundingMetadata | None,
        __metadata__: Metadata,
    ) -> list[str] | None:
        """
        The Search Suggestions embeds of the response so far (one per grounded round, at most five), or None
        if this round has none.
        """
        entry_point = grounding_metadata.search_entry_point if grounding_metadata else None
        if not (entry_point and entry_point.rendered_content):
            return None
        embed = _search_suggestions_embed(entry_point.rendered_content)
        state = self._turn_state(__metadata__)
        if state is None:
            return [embed]
        if embed not in state.suggestions and len(state.suggestions) < 5:
            state.suggestions.append(embed)
        return list(state.suggestions)

    @staticmethod
    def _finish_reason_name(response: types.GenerateContentResponse | None) -> str:
        """The finish reason of a response's first candidate as a name ('' if there is none)."""
        candidate = response.candidates[0] if response and response.candidates else None
        reason = candidate.finish_reason if candidate else None
        return str(getattr(reason, 'name', None) or reason or '')

    async def _process_part(
        self,
        part: types.Part,
        app: FastAPI,  # We need the app to generate URLs for model returned images.
        __metadata__: Metadata,
    ) -> tuple[dict | None, int]:
        """
        Processes a single `types.Part` object and returns a payload dictionary
        for the Open WebUI stream, along with a count of tag substitutions.
        The payload key is 'reasoning_content' for thought parts and 'content' for others.
        """
        # Determine the payload key based on whether the part is a thought.
        key = 'reasoning_content' if part.thought else 'content'
        payload_content: str | None = None
        count: int = 0

        match part:
            case types.Part(text=str(text)):
                # It's regular content or a thought with text.
                sanitized_text, count = self._disable_special_tags(text)
                payload_content = sanitized_text
            case types.Part(inline_data=data) if data:
                # An image part, which can be part of a thought or regular content.
                # Image parts don't need tag disabling.
                processed_text, image_url = await self._process_image_part(data, __metadata__, app)
                payload_content = processed_text

                # Transform inline_data into file_data to avoid storing raw bytes in the response history.
                # This mutates the part object which is held by reference in `response_parts`.
                if image_url and data.mime_type:
                    part.inline_data = None
                    part.file_data = types.FileData(file_uri=image_url, mime_type=data.mime_type)
            case types.Part(executable_code=code) if code:
                # Code blocks are already formatted and safe.
                if processed_text := self._process_executable_code_part(code):
                    payload_content = processed_text
            case types.Part(code_execution_result=result) if result:
                # Code results are also safe.
                if processed_text := self._process_code_execution_result_part(result):
                    payload_content = processed_text

        if payload_content is not None:
            return {key: payload_content}, count

        return None, 0

    @staticmethod
    def _disable_special_tags(text: str) -> tuple[str, int]:
        """
        Finds special tags in a text chunk and inserts a Zero-Width Space (ZWS)
        to prevent them from being parsed by the Open WebUI backend's legacy system.
        This is a safeguard against accidental tag generation by the model.
        """
        if not text:
            return '', 0

        # The regex finds '<' followed by an optional '/' and then one of the special tags.
        # The inner parentheses group the tags, so the optional '/' applies to all of them.
        TAG_REGEX = re.compile(r'<(/?' + '(' + '|'.join(re.escape(tag) for tag in SPECIAL_TAGS_TO_DISABLE) + ')' + r')')
        # The substitution injects a ZWS, e.g., '</think>' becomes '<ZWS/think'.
        modified_text, num_substitutions = TAG_REGEX.subn(rf'<{ZWS}\1', text)
        return modified_text, num_substitutions

    async def _process_image_part(
        self,
        inline_data: types.Blob,
        __metadata__: Metadata,
        app: FastAPI,
    ) -> tuple[str, str | None]:
        """
        Handles image data by saving it to the Open WebUI backend and returning a markdown link
        and the URL.
        """
        mime_type = inline_data.mime_type
        image_data = inline_data.data
        image_url = None

        if mime_type and image_data:
            image_url, _ = await self._upload_media(
                image_data,
                mime_type,
                __metadata__,
                app,
                'generated-image',
            )
        else:
            log.warning('Image part has no mime_type or data, cannot upload image. Returning a placeholder message.')

        markdown_text = (
            _image_markdown(image_url)
            if image_url
            else '*An error occurred while trying to store this model generated image.*'
        )
        return markdown_text, image_url

    async def _upload_media(
        self,
        data: bytes,
        mime_type: str,
        __metadata__: Metadata,
        app: FastAPI,
        name: str,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """
        Uploads model-generated media to the configured Open WebUI storage provider and links it to the chat.
        Returns the url to the uploaded file and its chat attachment entry.
        """
        extension = mimetypes.guess_extension(mime_type) or ('.png' if mime_type.startswith('image/') else '.bin')
        id = str(uuid.uuid4())
        filename = f'{name}{extension}'

        # The final filename includes the unique ID to prevent collisions.
        stored_name = f'{id}_{filename}'

        # Create a clean, precise metadata object linking to the generation context.
        media_metadata = {
            'model': __metadata__.get('canonical_model_id'),
            'chat_id': __metadata__.get('chat_id'),
            'message_id': __metadata__.get('message_id'),
        }

        log.info(f'Uploading the model-generated {mime_type} file to the Open WebUI backend.')

        try:
            contents, file_path = await asyncio.to_thread(Storage.upload_file, io.BytesIO(data), stored_name, tags={})
        except Exception:
            log.exception('Error occurred during upload to the storage provider.')
            return None, None

        log.debug('Adding the file to the Open WebUI files database.')
        file_item = await Files.insert_new_file(
            __metadata__.get('user_id'),
            FileForm(
                id=id,
                filename=filename,
                path=file_path,
                meta={
                    'name': filename,
                    'content_type': mime_type,
                    'size': len(contents),
                    'data': media_metadata,
                },
            ),
        )
        if not file_item:
            log.warning('File upload to Open WebUI database likely failed.')
            return None, None

        url: str = app.url_path_for('get_file_content_by_id', id=file_item.id)
        chat_id, message_id, user_id = (__metadata__.get(key) for key in ('chat_id', 'message_id', 'user_id'))
        if chat_id and message_id and user_id and not self._is_unsaved_chat(chat_id):
            # Like Open WebUI's own generated files: participants of a shared chat may then read it too.
            try:
                await Chats.insert_chat_files(chat_id, message_id, [file_item.id], user_id)
            except Exception:
                log.exception('Linking the generated file to the chat failed.')

        log.success('Media upload finished!')
        entry = {
            'id': file_item.id,
            'type': 'image' if mime_type.startswith('image/') else 'file',
            'url': url,
            'name': filename,
            'size': len(contents),
            'content_type': mime_type,
            'meta': {'content_type': mime_type, 'size': len(contents)},
        }
        return url, entry

    @staticmethod
    def _is_unsaved_chat(chat_id: str) -> bool:
        """Temporary (and channel) chats have no chat entry in the database."""
        return not chat_id or chat_id.startswith(('local', 'temporary', 'channel:'))

    @staticmethod
    def _process_executable_code_part(executable_code_part: types.ExecutableCode | None) -> str | None:
        """
        Processes an executable code part and returns the formatted string representation.
        """

        if not executable_code_part:
            return None

        lang_name = 'python'  # Default language
        if executable_code_part_lang_enum := executable_code_part.language:
            if lang_name := executable_code_part_lang_enum.name:
                lang_name = executable_code_part_lang_enum.name.lower()
            else:
                log.warning(
                    f'Could not extract language name from {executable_code_part_lang_enum}. Default to python.'
                )
        else:
            log.warning('Language Enum is None, defaulting to python.')

        if executable_code_part_code := executable_code_part.code:
            return f'```{lang_name}\n{executable_code_part_code.rstrip()}\n```\n\n'
        return ''

    @staticmethod
    def _process_code_execution_result_part(code_execution_result_part: types.CodeExecutionResult | None) -> str | None:
        """
        Processes a code execution result part and returns the formatted string representation.
        """

        if not code_execution_result_part:
            return None

        if code_execution_result_part_output := code_execution_result_part.output:
            return f'**Output:**\n\n```\n{code_execution_result_part_output.rstrip()}\n```\n\n'
        else:
            return None

    # endregion 1.4 Model response processing

    # region 1.5 Post-processing
    async def _do_post_processing(
        self,
        model_response: types.GenerateContentResponse | None,
        event_emitter: EventEmitter,
        app_state: State,
        __metadata__: Metadata,
        *,
        stream_error_happened: bool = False,
        final_round: bool = True,
    ):
        """Handles grounding and sources after the main response/stream is done."""
        log.info('Post-processing the model response.')

        if stream_error_happened:
            log.warning('Response processing failed due to stream error.')
            event_emitter.emit_status('Response failed [Stream Error]', done=True)
            return

        if not model_response:
            log.warning('Response processing skipped: Model response was empty.')
            event_emitter.emit_status('Response failed [Empty Response]', done=True)
            return

        if not (candidate := self._get_first_candidate(model_response.candidates)):
            log.warning('Response processing skipped: No candidates found.')
            event_emitter.emit_status('Response failed [No Candidates]', done=True)
            return

        # --- Construct detailed finish reason message ---
        reason_name = getattr(candidate.finish_reason, 'name', 'UNSPECIFIED')
        reason_description = FINISH_REASON_DESCRIPTIONS.get(reason_name)
        finish_message = candidate.finish_message.strip() if candidate.finish_message else None

        details_parts = [part for part in (reason_description, finish_message) if part]
        details_str = f': {" ".join(details_parts)}' if details_parts else ''
        full_finish_details = f'[{reason_name}]{details_str}'

        # --- Determine final status and emit toast for errors ---
        is_normal_finish = candidate.finish_reason in NORMAL_REASONS

        if is_normal_finish:
            log.debug(f'Response finished normally. {full_finish_details}')
            status_prefix = 'Response finished' if final_round else 'Tool calls requested'
        else:
            log.error(f'Response finished with an error. {full_finish_details}')
            status_prefix = 'Response failed'
            event_emitter.emit_toast(
                f'An error occurred. {full_finish_details}',
                'error',
            )

        # For the most common success case (STOP), we don't need to show the reason.
        final_reason_str = '' if reason_name == 'STOP' else f' [{reason_name}]'
        event_emitter.emit_status(
            f'{status_prefix}{final_reason_str}',
            done=True,
            is_successful_finish=is_normal_finish,
        )

        # TODO: Emit a toast message if url context retrieval was not successful.

        if final_round:
            # The companion filter's outlet adds citations to the text of the final round (the last one in
            # Open WebUI's tool loop), so only that round's grounding is stored; None clears an earlier one.
            self._store_data_in_state(app_state, __metadata__, {'grounding': candidate.grounding_metadata})

    @staticmethod
    def _calculate_cost(token_count: int, pricing_tiers: list[dict], prompt_tokens: int) -> float:
        """
        Calculates cost based on tiered pricing structure (in USD).

        Gemini prices depend on the prompt size: a prompt longer than a tier's `up_to_tokens` bills all
        tokens of the request at the next tier's price.
        """
        if not pricing_tiers or token_count <= 0:
            return 0.0

        for tier in pricing_tiers:
            tier_limit = tier.get('up_to_tokens')  # None means unlimited
            if tier_limit is None or prompt_tokens <= tier_limit:
                return (token_count / 1_000_000) * tier.get('price_per_million', 0.0)
        return (token_count / 1_000_000) * pricing_tiers[-1].get('price_per_million', 0.0)

    def _get_usage_data(
        self,
        response: types.GenerateContentResponse,
        metadata: Metadata,
        start_time: float,
        api_request_start_time: float | None = None,
        first_token_time: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Extracts usage data from a GenerateContentResponse object in the shape Open WebUI stores:
        `input_tokens` (prompt and tool-use prompt, cached tokens included), `output_tokens` (answer and
        thinking, which is billed as output) and `total_tokens`, the OpenAI-style aliases and details, and
        the cost from the model table's pricing. Open WebUI sums these over the rounds of its tool loop.
        Adds cumulative tokens and cost if previous history data is available.
        Returns None if usage metadata is not present.
        """
        if not response.usage_metadata:
            log.warning('Usage metadata is missing from the response. Cannot determine usage.')
            return None

        usage_metadata = response.usage_metadata
        # Dump the raw token usage details, excluding any fields that are None.
        token_details = usage_metadata.model_dump(exclude_none=True)

        prompt_tokens = usage_metadata.prompt_token_count or 0
        tool_prompt_tokens = usage_metadata.tool_use_prompt_token_count or 0
        cached_tokens = usage_metadata.cached_content_token_count or 0
        candidates_tokens = usage_metadata.candidates_token_count or 0
        thoughts_tokens = usage_metadata.thoughts_token_count or 0
        input_tokens = prompt_tokens + tool_prompt_tokens
        output_tokens = candidates_tokens + thoughts_tokens
        total_tokens = usage_metadata.total_token_count or input_tokens + output_tokens

        is_paid_api = metadata.get('is_paid_api', True)
        model_id = metadata.get('canonical_model_id', '')
        cost_details: dict[str, float] = {
            'input_cost': 0.0,
            'cache_cost': 0.0,
            'output_cost': 0.0,
            'image_output_cost': 0.0,
            'total_cost': 0.0,
        }

        pricing = self.catalog.entry(model_id).get('pricing') or {}
        if not is_paid_api:
            log.debug('Using free API, costs are not applicable and will be reported as 0.')
        elif not pricing:
            log.debug(f'No pricing data found for model {model_id}. Cost details will be empty.')
        else:
            image_tokens = sum(
                detail.token_count or 0
                for detail in usage_metadata.candidates_tokens_details or []
                if getattr(detail.modality, 'name', detail.modality) == 'IMAGE'
            )
            input_price = pricing.get('input', [])
            cost_details['input_cost'] = self._calculate_cost(input_tokens - cached_tokens, input_price, input_tokens)
            cost_details['cache_cost'] = self._calculate_cost(
                cached_tokens, pricing.get('caching', input_price), input_tokens
            )
            cost_details['output_cost'] = self._calculate_cost(
                output_tokens - image_tokens, pricing.get('output', []), input_tokens
            )
            cost_details['image_output_cost'] = self._calculate_cost(
                image_tokens, pricing.get('image_output', pricing.get('output', [])), input_tokens
            )
            cost_details['total_cost'] = sum(cost_details.values())
            cost_details = {key: round(value, 6) for key, value in cost_details.items()}
            log.debug(f'Calculated cost for model {model_id}:', payload=cost_details)

        usage_payload: dict[str, Any] = {
            # Open WebUI's canonical counters (summed over the tool loop's rounds) and their aliases.
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'prompt_tokens': input_tokens,
            'completion_tokens': output_tokens,
            'cache_read_input_tokens': cached_tokens,
            'prompt_tokens_details': {'cached_tokens': cached_tokens},
            'completion_tokens_details': {'reasoning_tokens': thoughts_tokens},
            'input_cost': round(cost_details['input_cost'] + cost_details['cache_cost'], 6),
            'output_cost': round(cost_details['output_cost'] + cost_details['image_output_cost'], 6),
            'total_cost': cost_details['total_cost'],
            # Gemini's own breakdown of this round.
            'token_details': token_details,
            'cost_details': cost_details,
        }

        # --- Calculate and append cumulative usage ---
        prev_tokens = metadata.get('cumulative_tokens')
        prev_cost = metadata.get('cumulative_cost')

        # Only add cumulative data if the chain is unbroken (previous data exists)
        if prev_tokens is not None and prev_cost is not None:
            # Open WebUI keeps the last round's value, so it covers all rounds of this response so far.
            state = self._turn_state(metadata)
            if state is not None:
                state.tokens += total_tokens
                state.cost += cost_details['total_cost']
            response_tokens, response_cost = (
                (state.tokens, state.cost) if state else (total_tokens, cost_details['total_cost'])
            )
            usage_payload['cumulative_token_count'] = prev_tokens + response_tokens
            usage_payload['cumulative_total_cost'] = round(prev_cost + response_cost, 6)

        now = time.monotonic()
        if api_request_start_time is not None and first_token_time is not None:
            usage_payload['time_to_first_token'] = round(first_token_time - api_request_start_time, 2)

        if api_request_start_time is not None:
            usage_payload['generation_time'] = round(now - api_request_start_time, 2)

        usage_payload['completion_time'] = round(now - start_time, 2)

        return usage_payload

    # endregion 1.5 Post-processing

    # region 1.6 __request__.app.state

    @staticmethod
    def _store_data_in_state(
        app_state: State,
        __metadata__: Metadata,
        data: dict[str, Any],
    ):
        """
        Stores multiple values in the app state, namespaced by chat and message ID; None removes a value.
        Exits early if this is a task model (e.g. title generation) to prevent
        state bloat and interference with the main chat's filter logic.
        """
        # TODO: code a separate dataclass that handles all stuff that I need to store in app state
        if __metadata__.get('task'):
            return

        chat_id = __metadata__.get('chat_id')
        message_id = __metadata__.get('message_id')

        if not chat_id or not message_id:
            log.warning('Skipping state storage: chat_id or message_id missing from metadata.')
            return

        for key_suffix, value in data.items():
            key = f'{key_suffix}_{chat_id}_{message_id}'
            log.debug(f"Storing data in app state with key '{key}'.")
            # Using shared `request.app.state` to pass data to Filter.outlet.
            # This is necessary because Pipe.pipe and Filter.outlet operate on different requests.
            if value is None:
                app_state._state.pop(key, None)
            else:
                app_state._state[key] = value

    @staticmethod
    def _get_and_clear_data_from_state(
        app_state: State,
        chat_id: str,
        message_id: str,
        key_suffix: str,
        clear_after_read: bool,
    ) -> Any | None:
        """Retrieves data from the app state using a namespaced key.

        Deletes the value only when clear_after_read is True.
        """
        key = f'{key_suffix}_{chat_id}_{message_id}'
        value = getattr(app_state, key, None)
        if value is None:
            return None

        if clear_after_read:
            log.debug(f"Retrieved and cleared data from app state for key '{key}'.")
            try:
                delattr(app_state, key)
            except AttributeError:
                # This case is unlikely but handles a race condition where the attribute might already be gone.
                log.warning(f"State key '{key}' was already gone before deletion attempt.")
        else:
            log.debug(f"Retrieved data from app state for key '{key}' without clearing it.")
        return value

    def _get_event_emitter(
        self,
        app_state: State,
        chat_id: str | None,
        message_id: str | None,
        is_task: bool,
    ) -> EventEmitter:
        """
        Resolves the event emitter that the companion filter's inlet stored for this request. Without the
        companion (or for background tasks) a silent emitter is used: errors still reach the chat as chunks.
        """
        dummy_emitter = app_state._state.get('gemini_dummy_event_emitter') or _SilentEmitter()
        if is_task:
            return dummy_emitter

        event_emitter = (
            self._get_and_clear_data_from_state(
                app_state,
                chat_id,
                message_id,
                key_suffix='gemini_event_emitter',
                clear_after_read=False,
            )
            if chat_id and message_id
            else None
        )
        if event_emitter is not None:
            return event_emitter

        log.warning(
            "No event emitter found in state for this request. Companion filter's inlet did not run? "
            'Status messages and toasts are disabled for this request.'
        )
        return dummy_emitter

    def _turn_state(self, __metadata__: Metadata) -> _TurnState | None:
        """
        The state of the response being generated, shared by the rounds of Open WebUI's tool loop (which
        call the pipe once per round with the same message id). None for tasks and requests without ids.
        """
        chat_id, message_id = __metadata__.get('chat_id'), __metadata__.get('message_id')
        if __metadata__.get('task') or not chat_id or not message_id:
            return None
        now = time.monotonic()
        for key in [k for k, state in self._turns.items() if now - state.updated > _TURN_STATE_TTL]:
            del self._turns[key]
        state = self._turns.setdefault(f'{chat_id}:{message_id}', _TurnState())
        state.updated = now
        return state

    # endregion 1.6 __request__.app.state

    # region 1.7 Utility helpers

    async def _determine_execution_order(
        self,
        options: ResolvedOptions,
        __metadata__: Metadata,
        features: Features,
    ) -> list[str]:
        """
        Calculates the sequence of execution tiers (free, paid, vertex).
        Returns an empty list if no valid routing configuration is found.
        """
        model_id = __metadata__.get('canonical_model_id', '')
        has_free_key = bool(options.GEMINI_FREE_API_KEY)
        has_paid_key = bool(options.GEMINI_PAID_API_KEY)

        # Retrieve Toggle Statuses
        vertex_available, vertex_toggled_on = await self._get_toggleable_feature_status(
            'gemini_vertex_ai_toggle', __metadata__
        )
        # Vertex is only viable if toggled on AND we have a project ID
        can_use_vertex = vertex_available and vertex_toggled_on and bool(options.VERTEX_PROJECT)
        # Maps grounding is not available on the free tier of most models.
        maps_available, maps_toggled_on = await self._get_toggleable_feature_status(
            'gemini_maps_grounding_toggle', __metadata__
        )

        (
            paid_toggle_available,
            paid_toggled_on,
        ) = await self._get_toggleable_feature_status('gemini_paid_api', __metadata__)

        task_type = __metadata__.get('task')
        routing_strategy = options.TASK_MODEL_ROUTING if task_type else 'match_main'

        is_free_eligible = self._check_free_tier_eligibility(
            model_id, features, maps_requested=maps_available and maps_toggled_on
        )

        execution_order: list[str] = []

        match routing_strategy:
            case 'only_free':
                log.debug('Task model routing override: only_free')
                if has_free_key:
                    if not is_free_eligible:
                        log.warning(
                            f"Task model '{model_id}' forced to 'only_free' but is ineligible. "
                            'Expect an upstream API error.'
                        )
                    execution_order = ['free']
                else:
                    log.error("Routing strategy 'only_free' requested, but no Free API Key is configured.")

            case 'free_fallback':
                log.debug('Task model routing override: free_fallback')
                # 1. Attempt Free if keys exist and model is eligible
                if has_free_key and is_free_eligible:
                    execution_order.append('free')

                # 2. Add Paid fallbacks
                if can_use_vertex:
                    execution_order.append('vertex')
                elif has_paid_key:
                    execution_order.append('paid')

                if not execution_order:
                    log.warning("Strategy 'free_fallback' could not find any viable tiers (check keys/eligibility).")

            case 'only_paid':
                log.debug('Task model routing override: only_paid')
                if can_use_vertex:
                    execution_order = ['vertex']
                elif has_paid_key:
                    execution_order = ['paid']
                else:
                    log.error(
                        "Routing strategy 'only_paid' requested, but neither Vertex AI nor Paid API is configured."
                    )

            case _:
                # Default Routing Logic ("match_main")
                if can_use_vertex:
                    execution_order = ['vertex']
                elif paid_toggle_available and paid_toggled_on:
                    if has_paid_key:
                        execution_order = ['paid']
                    else:
                        log.error('Paid API toggle is ON, but GEMINI_PAID_API_KEY is missing.')
                else:
                    # Logic for standard/un-toggled flow
                    if has_free_key:
                        if is_free_eligible:
                            execution_order = ['free']
                            if options.ENABLE_FREE_TIER_FALLBACK:
                                if can_use_vertex:
                                    execution_order.append('vertex')
                                elif has_paid_key:
                                    execution_order.append('paid')
                        else:
                            # FIXME: Does it make sense if paid tier fallback is explicitly blocked?
                            # If model isn't free-eligible, jump straight to paid tiers.
                            # If no paid tier exists, we try free anyway to let the API return the specific error.
                            if can_use_vertex:
                                execution_order = ['vertex']
                            elif has_paid_key:
                                execution_order = ['paid']
                            else:
                                log.warning(f"Model '{model_id}' is ineligible for free tier and no paid keys found.")
                    elif can_use_vertex:
                        execution_order = ['vertex']
                    elif has_paid_key:
                        execution_order = ['paid']

        log.debug(f'Routing strategy for {model_id} ({routing_strategy}): {execution_order}')
        return execution_order

    @staticmethod
    async def _get_toggleable_feature_status(
        filter_id: str,
        __metadata__: Metadata,
    ) -> tuple[bool, bool]:
        """
        Checks the complete status of a toggleable filter (function).

        This function performs a series of checks to determine if a feature
        is available for use and if the user has activated it.

        1. Checks if the filter is installed.
        2. Checks if the filter's master toggle is active in the Functions dashboard.
        3. Checks if the filter is enabled for the current model (or is global).
        4. Checks if the user has toggled the feature ON for the current request.

        Args:
            filter_id: The ID of the filter to check.
            __metadata__: The metadata object for the current request.

        Returns:
            A tuple (is_available: bool, is_toggled_on: bool).
            - is_available: True if the filter is installed, active, and configured for the model.
            - is_toggled_on: True if the user has the toggle ON in the UI for this request.
        """
        # 1. Check if the filter is installed
        f = await Functions.get_function_by_id(filter_id)
        if not f:
            log.warning(
                f"The '{filter_id}' filter is not installed. Install it to use the corresponding front-end toggle."
            )
            return (False, False)

        # 2. Check if the master toggle is active
        if not f.is_active:
            log.warning(
                f"The '{filter_id}' filter is installed but is currently disabled in the "
                'Functions dashboard (master toggle is off). Enable it to make it available.'
            )
            return (False, False)

        # 3. Check if the filter is enabled for the model or is global
        model_info = __metadata__.get('model', {}).get('info', {})
        model_filter_ids = model_info.get('meta', {}).get('filterIds', [])
        is_enabled_for_model = filter_id in model_filter_ids or f.is_global

        log.debug(
            f"Checking model enablement for '{filter_id}': in_model_filters={filter_id in model_filter_ids}, "
            f'is_global={f.is_global} -> is_enabled={is_enabled_for_model}'
        )

        if not is_enabled_for_model:
            # This is a configuration issue, not a user-facing warning. Debug is appropriate.
            model_id = __metadata__.get('model', {}).get('id', 'Unknown')
            log.debug(f"Filter '{filter_id}' is not enabled for model '{model_id}' and is not global.")
            return (False, False)

        # 4. Check if the user has toggled the feature ON for this request
        user_toggled_ids = __metadata__.get('filter_ids', [])
        is_toggled_on = filter_id in user_toggled_ids

        if is_toggled_on:
            log.info(f"Feature '{filter_id}' is available and enabled by the front-end toggle for this request.")
        else:
            log.debug(f"Feature '{filter_id}' is available but not enabled by the front-end toggle for this request.")

        return (True, is_toggled_on)

    def _get_first_candidate(self, candidates: list[types.Candidate] | None) -> types.Candidate | None:
        """Selects the first candidate, logging a warning if multiple exist."""
        if not candidates:
            # Logging warnings is handled downstream.
            return None
        if len(candidates) > 1:
            log.warning('Multiple candidates found, defaulting to first candidate.')
        return candidates[0]

    def _check_companion_filter_version(self, features: Features | dict) -> None:
        """
        Checks for the presence and version compatibility of the Gemini Manifold Companion filter.
        Logs warnings if the filter is missing or outdated.
        """
        companion_version = features.get('gemini_manifold_companion_version')

        if companion_version is None:
            log.warning(
                'Gemini Manifold Companion filter not detected. Without it there are no status messages,'
                ' toasts, citations, Google Search/code execution interception or RAG bypass. Install version'
                f' {RECOMMENDED_COMPANION_VERSION} or later and make sure it is active for this model'
                ' (or activate it globally).'
            )
            return

        # Comparing tuples of integers is a robust way to handle versions like '1.10.0' vs '1.2.0'.
        companion_match = re.match(r'(\d+)\.(\d+)\.(\d+)', str(companion_version))
        if not companion_match:
            # This handles cases where the version string is malformed (e.g., '1.a.0').
            log.error(f"Could not parse companion version string: '{companion_version}'. Version check skipped.")
            return
        companion_v_tuple = tuple(map(int, companion_match.groups()))
        recommended_v_tuple = tuple(map(int, RECOMMENDED_COMPANION_VERSION.split('.')))

        if companion_v_tuple < recommended_v_tuple:
            log.warning(
                f'The installed Gemini Manifold Companion filter version ({companion_version}) is older than '
                f'the recommended version ({RECOMMENDED_COMPANION_VERSION}). '
                'Some features may not work as expected. Please update the filter.'
            )
        else:
            log.debug(f'Gemini Manifold Companion filter detected with version: {companion_version}')

    def _interactions_models(self) -> list[ModelData]:
        """The Interactions API models enabled by the admin valves (virtual models, not in models.list)."""
        models: list[ModelData] = []
        if self.valves.ENABLE_DEEP_RESEARCH:
            models += [
                {
                    'id': agent_id,
                    'name': name,
                    'description': 'Autonomous multi-step web research ending in a cited report (background task).',
                }
                for agent_id, name in DEEP_RESEARCH_AGENTS.items()
            ]
        if self.valves.ENABLE_OMNI_VIDEO:
            models.append(
                {
                    'id': OMNI_VIDEO_MODEL,
                    'name': 'Gemini Omni 1.2 Flash (video)',
                    'description': 'Generates, edits and extends videos with native audio (Interactions API).',
                }
            )
        return models

    def _is_interactions_model(self, model_id: str, client: genai.Client) -> bool:
        """Whether the model runs on the Interactions API; raises if it is not enabled or not reachable."""
        if model_id not in DEEP_RESEARCH_AGENTS and model_id != OMNI_VIDEO_MODEL:
            return False
        if model_id not in {model['id'] for model in self._interactions_models()}:
            # They are billed per task or per second of video, so only the admin valves enable them.
            raise ValueError(f"'{model_id}' is not enabled (see ENABLE_DEEP_RESEARCH and ENABLE_OMNI_VIDEO).")
        if client.vertexai:
            raise ValueError(f"'{model_id}' runs on the Interactions API, which needs the Gemini Developer API.")
        return True

    def _accepts_functions(self, model_id: str) -> bool:
        """Whether Open WebUI's tools may be declared to the model (image models never take them)."""
        return not self.catalog.is_image_model(model_id) and self.catalog.capability(model_id, 'function_calling', True)

    def _image_media_resolution(self, options: ResolvedOptions, model_id: str) -> types.PartMediaResolution | None:
        """`ULTRA_HIGH` exists for images only, as a per-part resolution of Gemini 3 and later."""
        if options.MEDIA_RESOLUTION != 'ULTRA_HIGH' or self._parse_gemini_version(model_id) < 3.0:
            return None
        return types.PartMediaResolution(level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH)

    # endregion 1.7 Utility helpers

    # endregion 1. Helper methods inside the Pipe class


# region Option resolution

OPTION_SYNONYMS: Final[dict[str, str]] = {
    'reasoning_effort': 'THINKING_CONFIG_RULES',
    'thinking_budget': 'THINKING_CONFIG_RULES',
    'thinking_level': 'THINKING_CONFIG_RULES',
}

# Dynamic sets distinguishing overridable vs admin-only valve options
USER_OVERRIDABLE_VALVE_FIELDS: Final[set[str]] = set(_SharedValves.model_fields.keys())
ADMIN_ONLY_VALVE_FIELDS: Final[set[str]] = set(Pipe.Valves.model_fields.keys()) - USER_OVERRIDABLE_VALVE_FIELDS

# Pre-computed case-insensitive lookup dictionaries
_VALVE_FIELD_LOOKUP: Final[dict[str, str]] = {
    field_name.lower(): field_name for field_name in Pipe.Valves.model_fields.keys()
}

_SYNONYM_LOOKUP: Final[dict[str, str]] = {
    synonym.lower(): canonical_target for synonym, canonical_target in OPTION_SYNONYMS.items()
}


def canonicalize_option_key(key: str) -> str:
    """
    Normalizes an option key using case-insensitive lookup against synonyms
    and known valve fields. Returns the canonical key name or the stripped key if custom.
    """
    key_clean = key.strip()
    key_lower = key_clean.lower()

    if canonical := _SYNONYM_LOOKUP.get(key_lower):
        return canonical

    if canonical := _VALVE_FIELD_LOOKUP.get(key_lower):
        return canonical

    return key_clean


class ResolvedOptions(Pipe.Valves):
    """
    Consolidated configuration combining Admin Valves, User Valves,
    Model Advanced Params, and Chat Advanced Params according to priority.

    Allows extra dynamic options defined at model or chat levels.
    """

    model_config = ConfigDict(extra='allow')

    def get_custom_params(self) -> dict[str, Any]:
        """Returns extra parameters not defined in Pipe.Valves."""
        valve_fields = set(Pipe.Valves.model_fields.keys())
        return {k: v for k, v in self.model_dump().items() if k not in valve_fields}


def _resolve_options(
    admin_valves: Pipe.Valves,
    user_valves: _SharedValves | None,
    user_email: str,
    body: Body,
    metadata: Metadata,
) -> ResolvedOptions:
    """
    Open WebUI (v0.12+) has already merged all parameter sources into `body`
    by the time the pipe runs (global defaults < model page params < chat params,
    with body-level keys winning over model-side writes), so anything unknown in
    `body` is simply adopted as an option.

    Admin-only options cannot be overridden by user-level sources.
    Option keys and synonyms are normalized case-insensitively.
    An invalid user-level value (e.g. an unknown `reasoning_effort`) is logged and ignored.
    """
    # Priority 1: Base options from Admin Valves
    merged_data = admin_valves.model_dump()
    # Where each user-level override came from, for the warning if it turns out invalid.
    overrides: dict[str, str] = {}

    # Priority 2: User Valves (user-overridable fields only)
    if user_valves is not None:
        for field_name in USER_OVERRIDABLE_VALVE_FIELDS:
            user_val = getattr(user_valves, field_name, None)
            if user_val is not None and user_val != '':
                merged_data[field_name] = user_val
                overrides[field_name] = 'user valve'

    # Enforce admin authentication security rules
    user_whitelist = admin_valves.AUTH_WHITELIST.split(',') if admin_valves.AUTH_WHITELIST else []
    if admin_valves.USER_MUST_PROVIDE_AUTH_CONFIG and user_email not in user_whitelist:
        log.info(
            f"User '{user_email}' is required to provide their own authentication credentials"
            ' due to USER_MUST_PROVIDE_AUTH_CONFIG=True.'
        )
        merged_data['GEMINI_FREE_API_KEY'] = getattr(user_valves, 'GEMINI_FREE_API_KEY', None) if user_valves else None
        merged_data['GEMINI_PAID_API_KEY'] = getattr(user_valves, 'GEMINI_PAID_API_KEY', None) if user_valves else None
        merged_data['VERTEX_PROJECT'] = None
        merged_data['USE_VERTEX_AI'] = False

    # Priority 3: Anything that the backend merged into the body is treated as
    # an option override. Since Open WebUI v0.12 the backend itself resolves
    # the priority between admin defaults, model page params and chat params
    # before the pipe runs, so no per-source distinction is needed here.
    known_body_keys = {
        'stream',
        'model',
        'messages',
        'files',
        'options',
        'stream_options',
        'features',
        'metadata',
        'params',
        'tool_ids',
        'tools',
        'tool_choice',
        'response_format',
        'variables',
        'skill_ids',
        'terminal_id',
        'chat_id',
    }
    body_options = {k: v for k, v in body.items() if k not in known_body_keys}
    if isinstance(body.get('options'), dict):
        for opt_k, opt_v in body['options'].items():  # type: ignore[reportTypedDictNotRequiredAccess]
            if opt_k not in body_options:
                body_options[opt_k] = opt_v

    for raw_key, val in body_options.items():
        if val is None or val == '':
            continue
        canonical_key = canonicalize_option_key(raw_key)
        if canonical_key in ADMIN_ONLY_VALVE_FIELDS:
            log.warning(
                f"Body parameter '{raw_key}' attempts to override admin-only valve"
                f" '{canonical_key}'. Ignoring override."
            )
            continue
        merged_data[canonical_key] = val
        overrides[canonical_key] = f"request parameter '{raw_key}'"

    try:
        return ResolvedOptions(**merged_data)
    except ValidationError as e:
        return ResolvedOptions(**_without_invalid_overrides(e, merged_data, overrides, admin_valves))


def _without_invalid_overrides(
    error: ValidationError,
    merged_data: dict[str, Any],
    overrides: dict[str, str],
    admin_valves: Pipe.Valves,
) -> dict[str, Any]:
    """The merged options with each invalid user-level override replaced by the admin setting."""
    invalid = {str(detail['loc'][0]) for detail in error.errors() if detail.get('loc')} & overrides.keys()
    if not invalid:
        raise error
    admin_data = admin_valves.model_dump()
    for field_name in invalid:
        log.warning(
            f'Ignoring invalid {overrides[field_name]} for {field_name}: {merged_data[field_name]!r}.'
            ' Using the admin setting instead.'
        )
        merged_data[field_name] = admin_data.get(field_name)
    return merged_data


# endregion Option resolution


# region Native tool calling

# The tool name mapping, the JSON Schema clean-up and the conversion of tool rounds follow the approach of owndev's
# Google Gemini Pipeline 1.20.0 (https://github.com/owndev/Open-WebUI-Functions, Apache License 2.0), re-implemented
# here in this pipe's own structure.


class ToolBridge:
    """
    Translates between Open WebUI's native tool calling (OpenAI-style `body['tools']`, `tool_choice` and
    `tool_calls`) and Gemini function calling for one request. Open WebUI runs the calls and sends their
    results back in the next round of its tool loop, so the bridge only declares functions and converts calls.
    """

    # Gemini function names: a letter or underscore, then letters, digits, underscores or dashes (the API also
    # accepts dots and colons, which models then tend to confuse with their `default_api.` call prefix).
    _NAME_RE: Final = re.compile(r'[A-Za-z_][A-Za-z0-9_-]{0,63}')
    # JSON Schema keys that `parameters_json_schema` does not need or that are reported to fail.
    _SCHEMA_DROP_KEYS: Final = frozenset(
        {'$schema', '$id', '$comment', 'examples', 'deprecated', 'readOnly', 'writeOnly'}
    )
    # Keywords whose values map names to schemas (the names are not keywords themselves).
    _SCHEMA_NAME_MAP_KEYS: Final = frozenset(
        {'properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'}
    )

    def __init__(self, tools: Any = None, *, user_selected: bool = False):
        self.declarations: list[types.FunctionDeclaration] = []
        # Whether the request offers tools besides Open WebUI's built-in ones (worth a toast if they are dropped).
        self.user_selected = user_selected
        self._owui_names: dict[str, str] = {}

        for entry in tools if isinstance(tools, list) else []:
            if not isinstance(entry, dict) or entry.get('type', 'function') != 'function':
                continue
            spec = entry.get('function')
            name = spec.get('name') if isinstance(spec, dict) else None
            if not isinstance(name, str) or not name:
                continue
            gemini_name = self.gemini_name(name)
            if gemini_name in self._owui_names:
                # Gemini rejects a request with duplicate declarations.
                log.warning(f"Skipping duplicate tool declaration '{name}'.")
                continue
            self._owui_names[gemini_name] = name

            declaration: dict[str, Any] = {'name': gemini_name}
            if isinstance(description := spec.get('description'), str) and description.strip():
                declaration['description'] = description.strip()
            if (schema := self._parameters_schema(spec.get('parameters'))) is not None:
                declaration['parameters_json_schema'] = schema
            self.declarations.append(types.FunctionDeclaration(**declaration))

    @classmethod
    def for_request(cls, body: Body, __metadata__: Metadata, *, accepts_functions: bool) -> ToolBridge:
        """
        The tools of a request in Open WebUI's native function calling mode. There are none in legacy mode
        (Open WebUI picks and runs tools itself before calling the pipe), for models without function calling,
        and for background tasks except context compaction (which replays the chat's tools, see the config).
        """
        params = __metadata__.get('params') or {}
        task = __metadata__.get('task')
        if (
            params.get('function_calling') == 'legacy'
            or not accepts_functions
            or (task and task != 'context_compaction')
        ):
            return cls()
        tool_specs = __metadata__.get('tools')
        user_selected = isinstance(tool_specs, dict) and any(
            isinstance(tool, dict) and tool.get('type') != 'builtin' for tool in tool_specs.values()
        )
        return cls(body.get('tools'), user_selected=user_selected)

    @classmethod
    def gemini_name(cls, name: str) -> str:
        """
        Maps an Open WebUI tool name to a valid Gemini function name. Valid names stay as they are; others get
        their invalid characters replaced and a hash suffix, which keeps the mapping stateless across rounds.
        """
        if cls._NAME_RE.fullmatch(name):
            return name
        base = re.sub(r'[^A-Za-z0-9_-]', '_', name)
        if not re.match(r'[A-Za-z_]', base[:1] or '0'):
            base = f'_{base}'
        digest = hashlib.sha1(name.encode('utf-8')).hexdigest()[:8]
        return f'{base[:55]}_{digest}'

    def owui_name(self, gemini_name: str) -> str:
        """Maps a Gemini function call name back to the Open WebUI tool (unknown names stay as they are)."""
        for prefix in ('default_api.', 'default_api:'):
            if gemini_name.startswith(prefix) and gemini_name[len(prefix) :] in self._owui_names:
                gemini_name = gemini_name[len(prefix) :]
        return self._owui_names.get(gemini_name, gemini_name)

    def function_calling_config(self, tool_choice: Any) -> types.FunctionCallingConfig | None:
        """
        Maps an OpenAI `tool_choice` to Gemini's function calling mode. `auto` (or nothing) keeps the API
        default (AUTO, or VALIDATED when built-in tools are combined with functions). With NONE the
        declarations stay, so function calls in the history remain valid.
        """
        mode = types.FunctionCallingConfigMode
        if tool_choice == 'none':
            return types.FunctionCallingConfig(mode=mode.NONE)
        if tool_choice in ('required', 'any'):
            return types.FunctionCallingConfig(mode=mode.ANY)
        if isinstance(tool_choice, dict) and tool_choice.get('type') == 'function':
            name = (tool_choice.get('function') or {}).get('name')
            gemini_name = self.gemini_name(name) if isinstance(name, str) and name else None
            if gemini_name in self._owui_names:
                return types.FunctionCallingConfig(mode=mode.ANY, allowed_function_names=[gemini_name])
            log.debug(f"Ignoring tool_choice for undeclared function '{name}'.")
        return None

    def tool_calls(self, function_call_parts: list[types.Part]) -> list[dict[str, Any]]:
        """
        Converts the function calls of a round to OpenAI `tool_calls`, keeping Gemini's call ids (the
        function responses must echo them). Gemini 2.5 and Vertex AI may send none; an id is made up then.
        """
        tool_calls: list[dict[str, Any]] = []
        for index, part in enumerate(function_call_parts):
            call = cast(types.FunctionCall, part.function_call)
            tool_calls.append(
                {
                    'index': index,
                    'id': call.id or f'call_{uuid.uuid4().hex[:24]}',
                    'type': 'function',
                    'function': {
                        'name': self.owui_name(call.name or ''),
                        'arguments': json.dumps(call.args or {}, ensure_ascii=False, default=str),
                    },
                }
            )
        return tool_calls

    @classmethod
    def _parameters_schema(cls, schema: Any) -> dict[str, Any] | None:
        """
        Cleans up a tool's JSON Schema for `parameters_json_schema`, which accepts `$ref`/`$defs`, `anyOf`,
        type arrays and the like, unlike Gemini's OpenAPI-style `parameters`. Returns None for an object
        schema without properties: such a tool is declared without parameters.
        """
        root = cls._clean_schema(copy.deepcopy(schema) if isinstance(schema, dict) else {})
        root.setdefault('type', 'object')
        if (
            root.get('type') == 'object'
            and not root.get('properties')
            and not any(key in root for key in ('$ref', 'anyOf', 'oneOf', 'allOf'))
        ):
            return None
        return root

    @classmethod
    def _clean_schema(cls, node: Any) -> Any:
        """Drops the keys Gemini does not need from a JSON Schema node and its sub-schemas."""
        if isinstance(node, list):
            return [cls._clean_schema(item) for item in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in cls._SCHEMA_DROP_KEYS or str(key).startswith('x-'):
                continue
            if key in ('exclusiveMinimum', 'exclusiveMaximum') and isinstance(value, bool):
                continue  # Draft 4 booleans; later drafts use numbers.
            if key == 'items' and isinstance(value, list):
                out['prefixItems'] = cls._clean_schema(value)  # Tuple form.
            elif key in cls._SCHEMA_NAME_MAP_KEYS and isinstance(value, dict):
                out[key] = {name: cls._clean_schema(sub_schema) for name, sub_schema in value.items()}
            else:
                out[key] = cls._clean_schema(value)
        required, properties = out.get('required'), out.get('properties')
        if isinstance(required, list) and isinstance(properties, dict):
            out['required'] = [name for name in required if name in properties]
        return out


# endregion Native tool calling


# region Request and response helpers

_LEVEL_RANK: Final[dict[str, int]] = {name: rank for rank, name in enumerate(_THINKING_LEVELS)}
# Gemini 2.5 budgets of the `reasoning_effort` values in Gemini's OpenAI compatibility layer
# (aistudio.google.com/docs/openai#thinking).
_LEVEL_TO_BUDGET: Final[dict[str, int]] = {'MINIMAL': 1024, 'LOW': 1024, 'MEDIUM': 8192, 'HIGH': 24576, 'XHIGH': 24576}

# Citation markers that the companion filter inserts into the displayed text: [1], [1][2], ...
_CITATION_MARKERS_RE: Final = re.compile(r'(?:\[\d+\])+')
# The Google Maps attribution block that `_maps_attribution` appends to a round's text.
_MAPS_ATTRIBUTION_RE: Final = re.compile(r'\n*---\n(?:- [^\n]* · Google\u00a0Maps\n?)+\s*$')

# Seconds of inactivity after which the state of a response (see `Pipe._turn_state`) is dropped.
_TURN_STATE_TTL: Final[float] = 3600.0


def _nearest_level(level: str, levels: list[str]) -> str:
    """The supported thinking level closest to `level`; a tie rounds up (more thinking rather than less)."""
    rank = _LEVEL_RANK.get(level, _LEVEL_RANK['MEDIUM'])
    return min(levels, key=lambda name: (abs(_LEVEL_RANK.get(name, rank) - rank), -_LEVEL_RANK.get(name, rank)))


def _budget_to_level(budget: int) -> str:
    """The thinking level of a thinking budget, by the thresholds of `_LEVEL_TO_BUDGET`."""
    return 'LOW' if budget <= 1024 else 'MEDIUM' if budget <= 8192 else 'HIGH'


def _global_media_resolution(value: str | None) -> types.MediaResolution | None:
    """The request-wide media resolution (`ULTRA_HIGH` only exists per image part, so it means HIGH here)."""
    if not value:
        return None
    return types.MediaResolution[f'MEDIA_RESOLUTION_{"HIGH" if value == "ULTRA_HIGH" else value}']


def _compact_parts(parts: list[types.Part]) -> list[types.Part]:
    """
    The parts of a round as they are sent back in later requests: unsigned thought summaries (display only)
    and empty fragments dropped, consecutive plain text fragments of a stream merged. A part with a thought
    signature is never merged or changed (aistudio.google.com/docs/thought-signatures).
    """
    compact: list[types.Part] = []
    mergeable = False
    for part in parts:
        if part.thought and not part.thought_signature:
            continue
        plain = part.text is not None and set(part.model_dump(exclude_none=True)) <= {'text', 'thought'}
        if plain and not part.text:
            continue
        if plain and mergeable:
            compact[-1] = compact[-1].model_copy(update={'text': f'{compact[-1].text}{part.text}'})
        else:
            compact.append(part)
        mergeable = plain
    return compact


def _comparable_text(text: str) -> str:
    """
    Normalizes the visible text of an assistant message for comparison with its stored parts: re-enables the
    disabled special tags and drops the citation markers, Google Maps attributions and whitespace that were
    added to the displayed text.
    """
    text = GeminiContentBuilder._enable_special_tags(text)
    text = _MAPS_ATTRIBUTION_RE.sub('', text)
    text = _CITATION_MARKERS_RE.sub('', text)
    return ' '.join(text.split())


def _is_grounded(grounding_metadata: types.GroundingMetadata | None) -> bool:
    """Whether a round's answer is grounded (and therefore gets citations)."""
    return bool(grounding_metadata and (grounding_metadata.grounding_chunks or grounding_metadata.grounding_supports))


def _image_markdown(url: str) -> str:
    """How a generated image appears in the answer text."""
    return f'![Generated Image]({url})'


def _markdown_link(title: str, uri: str) -> str:
    """A Markdown link whose text and target cannot break out of the link syntax."""
    title = re.sub(r'([\\\[\]])', r'\\\1', ' '.join(title.split()))
    uri = uri.replace(' ', '%20').replace('(', '%28').replace(')', '%29')
    return f'[{title}]({uri})'


def _maps_attribution(grounding_metadata: types.GroundingMetadata | None) -> str | None:
    """
    The Google Maps sources of a round as a list right after its text. Google requires each source (and each
    review snippet) to be shown with its title, a link and the unmodified text "Google Maps", immediately
    after the grounded content (aistudio.google.com/docs/maps-grounding#display-sources). A no-break space
    keeps "Google Maps" on one line.
    """
    links: dict[str, str] = {}
    for chunk in grounding_metadata.grounding_chunks or [] if grounding_metadata else []:
        if not (maps := chunk.maps):
            continue
        if maps.uri:
            links.setdefault(maps.uri, maps.title or maps.uri)
        answer_sources = maps.place_answer_sources
        for snippet in (answer_sources.review_snippets or []) if answer_sources else []:
            if snippet.google_maps_uri:
                links.setdefault(snippet.google_maps_uri, snippet.title or maps.title or 'Review')
    if not links:
        return None
    return '\n\n---\n' + '\n'.join(f'- {_markdown_link(title, uri)} · Google\u00a0Maps' for uri, title in links.items())


def _camel_case_request(
    config: types.GenerateContentConfig, contents: list[types.Content]
) -> types.GenerateContentConfig:
    """
    The SDK writes some objects into the request body as they are, so their fields go out in snake_case
    (`parameters_json_schema`, `thinking_level`, `file_uri`). Google reads both spellings, but a proxy that
    parses the REST schema may read camelCase only and drop the rest; AxonHub's native Gemini endpoint does:
    tools arrive without parameters, the thinking config is lost and file parts lose their URI.

    The contents, the tools and the thinking config are therefore sent camelCase in `http_options.extra_body`.
    Lists there replace the SDK's lists as they are. The thinking config is taken out of the config itself:
    the SDK renames the keys of a dict merged into one it already wrote back to its own spelling.
    """

    def camel(model: BaseModel) -> dict[str, Any]:
        return model.model_dump(mode='json', by_alias=True, exclude_none=True)

    extra_body: dict[str, Any] = {'contents': [camel(content) for content in contents]}
    if config.tools:
        extra_body['tools'] = [camel(tool) for tool in config.tools]
    if config.thinking_config:
        extra_body['generationConfig'] = {'thinkingConfig': camel(config.thinking_config)}
    http_options = config.http_options.model_copy() if config.http_options else types.HttpOptions()
    http_options.extra_body = {**(http_options.extra_body or {}), **extra_body}
    return config.model_copy(update={'thinking_config': None, 'http_options': http_options})


def _search_suggestions_embed(rendered_content: str) -> str:
    """
    Wraps Google's Search Suggestions (`searchEntryPoint.renderedContent`), unmodified, in a document for
    Open WebUI's sandboxed embed frame: links open in a new tab (no interstitials), and the frame reports the
    height of the chips so that it fits them (aistudio.google.com/docs/terms#grounding-with-google-search).
    The body is measured, not the document: inside a frame the document is never shorter than the frame
    (150px until it is resized), so the frame could never shrink. The 1px padding keeps the outline visible.
    """
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><base target="_blank">'
        '<style>html,body{margin:0;background:transparent}body{padding:1px;display:flow-root}</style>'
        '</head><body>'
        f'{rendered_content}'
        '<script>(()=>{const post=()=>parent.postMessage({type:"iframe:height",'
        'height:Math.ceil(document.body.getBoundingClientRect().height)},"*");'
        'addEventListener("load",post);new ResizeObserver(post).observe(document.body);})();</script>'
        '</body></html>'
    )


def _previous_interaction_id(messages: list[Message]) -> str | None:
    """The interaction that produced the last assistant message (Interactions API models), if any."""
    for message in reversed(messages):
        if message.get('role') != 'assistant':
            continue
        for item in message.get('reasoning_details') or []:
            if isinstance(item, dict) and item.get('format') == INTERACTION_FORMAT and item.get('data'):
                return str(item['data'])
        return None
    return None


@dataclass
class _TurnState:
    """What the rounds of one response share in Open WebUI's tool loop."""

    tokens: int = 0
    cost: float = 0.0
    suggestions: list[str] = field(default_factory=list)
    updated: float = 0.0


class _SilentEmitter:
    """Stands in for the companion filter's EventEmitter when the filter did not run: shows nothing."""

    def __init__(self) -> None:
        self.start_time = time.monotonic()

    def emit_toast(self, msg: str, type: str = 'info') -> None:
        pass

    def emit_status(self, description: str, done: bool = False, hidden: bool = False, **_: Any) -> None:
        pass


# endregion Request and response helpers


# region Interactions API models


class InteractionsRunner:
    """
    Runs models that only exist on the Interactions API as chat models: the Deep Research agents (a background
    task polled until it finishes) and Omni video generation. The interaction id is kept in the response's
    `reasoning_details`, so a follow-up message continues the interaction (a refined report, an edited or
    extended video).
    """

    POLL_INTERVAL: Final[float] = 10.0
    # Deep Research tasks take up to 60 minutes.
    DEEP_RESEARCH_TIMEOUT: Final[float] = 70 * 60.0
    FILE_ACTIVE_TIMEOUT: Final[float] = 120.0

    def __init__(
        self,
        *,
        client: genai.Client,
        model_id: str,
        options: ResolvedOptions,
        event_emitter: EventEmitter,
        upload: Any,
    ):
        self.client = client
        self.model_id = model_id
        self.options = options
        self.event_emitter = event_emitter
        # `upload(data, mime_type, name)` stores generated media in Open WebUI -> (url, chat file entry).
        self.upload = upload

    async def run(
        self,
        contents: list[types.Content],
        previous_interaction_id: str | None,
    ) -> AsyncGenerator[dict, None]:
        """Yields the response chunks of one interaction, or an error chunk."""
        user_input = self._input_blocks(contents)
        if not user_input:
            yield {'error': {'detail': 'There is no message to send.'}}
            return
        try:
            if self.model_id in DEEP_RESEARCH_AGENTS:
                async for chunk in self._deep_research(user_input, previous_interaction_id):
                    yield chunk
            else:
                async for chunk in self._omni_video(user_input, previous_interaction_id):
                    yield chunk
        except Exception as e:
            log.exception(f'Interaction with {self.model_id} failed.')
            self.event_emitter.emit_status('Response failed', done=True)
            yield {'error': {'detail': f'{self.model_id} failed: {e}'}}

    async def _deep_research(
        self, user_input: list[dict[str, Any]], previous_id: str | None
    ) -> AsyncGenerator[dict, None]:
        """Starts a research task in the background and polls it, showing its thought summaries."""
        request: dict[str, Any] = {
            'agent': self.model_id,
            'input': user_input,
            'background': True,
            'store': True,
            'agent_config': {'type': 'deep-research', 'thinking_summaries': 'auto'},
        }
        if previous_id:
            request['previous_interaction_id'] = previous_id
        self.event_emitter.emit_status('Starting Deep Research...')
        interaction = await self.client.aio.interactions.create(**request)
        yield self._interaction_detail(interaction.id)

        started = time.monotonic()
        shown_thoughts = 0
        try:
            while interaction.status in ('in_progress', 'queued'):
                elapsed = time.monotonic() - started
                if elapsed > self.DEEP_RESEARCH_TIMEOUT:
                    raise TimeoutError(
                        f'the research did not finish within {self.DEEP_RESEARCH_TIMEOUT / 60:.0f} minutes'
                    )
                self.event_emitter.emit_status(f'Deep Research in progress ({elapsed / 60:.0f} min)...')
                await asyncio.sleep(self.POLL_INTERVAL)
                interaction = await self.client.aio.interactions.get(id=interaction.id)
                thoughts = self._thought_summaries(interaction)
                for text in thoughts[shown_thoughts:]:
                    yield {'choices': [{'delta': {'reasoning_content': f'{text}\n\n'}}]}
                shown_thoughts = len(thoughts)
        except BaseException:
            # Stopped by the user, timed out or failed while polling: do not leave the task running (and billing).
            await self._cancel(interaction.id)
            raise

        if interaction.status != 'completed':
            raise RuntimeError(
                f'the research ended with status {interaction.status} {interaction.errors or ""}'.strip()
            )
        for chunk in await self._report_chunks(interaction):
            yield chunk
        self.event_emitter.emit_status('Deep Research finished', done=True)

    async def _report_chunks(self, interaction: Any) -> list[dict[str, Any]]:
        """The final report of a research task with its charts, its cited sources and the usage."""
        report, sources, images = self._report(interaction)
        for data, mime_type in images:
            url, _ = await self.upload(data, mime_type, 'deep-research-chart')
            if url:
                report += f'\n\n![Chart]({url})'
        chunks: list[dict[str, Any]] = [{'choices': [{'delta': {'content': report}}]}]
        chunks += [{'event': {'type': 'source', 'data': source}} for source in sources]
        if usage := self._usage(interaction):
            chunks.append({'usage': usage})
        return chunks

    async def _omni_video(
        self, user_input: list[dict[str, Any]], previous_id: str | None
    ) -> AsyncGenerator[dict, None]:
        """Generates (or, as a follow-up, edits or extends) a video and attaches it to the message."""
        response_format: dict[str, Any] = {'type': 'video', 'delivery': 'uri'}
        duration = self.options.OMNI_VIDEO_DURATION
        for key, value in (
            ('aspect_ratio', self.options.OMNI_VIDEO_ASPECT_RATIO),
            ('resolution', self.options.OMNI_VIDEO_RESOLUTION),
            ('duration', f'{duration}s' if duration else None),
        ):
            if value:
                response_format[key] = value
        request: dict[str, Any] = {
            'model': OMNI_VIDEO_MODEL,
            'input': user_input,
            'response_format': response_format,
            'store': True,
        }
        if previous_id:
            request['previous_interaction_id'] = previous_id
        self.event_emitter.emit_status('Generating video...')
        interaction = await self.client.aio.interactions.create(**request)
        if interaction.status != 'completed':
            raise RuntimeError(f'the video ended with status {interaction.status} {interaction.errors or ""}'.strip())

        video = interaction.output_video
        if video is None or not (video.data or video.uri):
            raise RuntimeError('the response holds no video')
        data = base64.b64decode(video.data) if video.data else await self._download(cast(str, video.uri))
        url, entry = await self.upload(data, video.mime_type or 'video/mp4', 'generated-video')
        if not url:
            raise RuntimeError('the video could not be stored in Open WebUI')

        yield self._interaction_detail(interaction.id)
        # Open WebUI renders a <video> element whose text is the source URL as a video player.
        yield {'choices': [{'delta': {'content': f'<video>{url}</video>'}}]}
        yield {'event': {'type': 'files', 'data': {'files': [entry]}}}
        self.event_emitter.emit_status('Video generated', done=True)
        if usage := self._usage(interaction):
            yield {'usage': usage}

    async def _download(self, uri: str) -> bytes:
        """Downloads a video delivered as a Files API file once the file is active."""
        if not (match := re.search(r'files/([^/:?]+)', uri)):
            raise RuntimeError(f'unexpected video URI {uri}')
        name = f'files/{match.group(1)}'
        deadline = time.monotonic() + self.FILE_ACTIVE_TIMEOUT
        while (file := await self.client.aio.files.get(name=name)).state != types.FileState.ACTIVE:
            if file.state == types.FileState.FAILED or time.monotonic() > deadline:
                raise RuntimeError(f'the video file {name} did not become available')
            await asyncio.sleep(2)
        return cast(bytes, await self.client.aio.files.download(file=name))

    async def _cancel(self, interaction_id: str) -> None:
        try:
            await self.client.aio.interactions.cancel(id=interaction_id)
            log.info(f'Cancelled interaction {interaction_id}.')
        except Exception:
            log.exception(f'Cancelling interaction {interaction_id} failed.')

    @staticmethod
    def _interaction_detail(interaction_id: str) -> dict[str, Any]:
        detail = {'type': 'reasoning.encrypted', 'format': INTERACTION_FORMAT, 'index': 0, 'data': interaction_id}
        return {'choices': [{'delta': {'reasoning_details': [detail]}}]}

    @staticmethod
    def _input_blocks(contents: list[types.Content]) -> list[dict[str, Any]]:
        """The last user turn as Interactions API input blocks (earlier turns live in the previous interaction)."""
        last = next((content for content in reversed(contents) if content.role == 'user'), None)
        blocks: list[dict[str, Any]] = []
        for part in (last.parts or []) if last else []:
            if part.text:
                blocks.append({'type': 'text', 'text': part.text})
            elif (media := part.inline_data or part.file_data) and media.mime_type:
                kind = next((k for k in ('image', 'video', 'audio') if media.mime_type.startswith(f'{k}/')), 'document')
                block: dict[str, Any] = {'type': kind, 'mime_type': media.mime_type}
                if part.inline_data:
                    block['data'] = base64.b64encode(part.inline_data.data or b'').decode('ascii')
                else:
                    block['uri'] = cast(types.FileData, part.file_data).file_uri
                blocks.append(block)
        return blocks

    @staticmethod
    def _thought_summaries(interaction: Any) -> list[str]:
        return [
            text
            for step in interaction.steps or []
            if getattr(step, 'type', None) == 'thought'
            for item in getattr(step, 'summary', None) or []
            if (text := getattr(item, 'text', None))
        ]

    @staticmethod
    def _report(interaction: Any) -> tuple[str, list[dict[str, Any]], list[tuple[bytes, str]]]:
        """The final report of a research task: its text, its cited sources and its charts."""
        outputs = [step for step in interaction.steps or [] if getattr(step, 'type', None) == 'model_output']
        texts: list[str] = []
        sources: dict[str, dict[str, Any]] = {}
        images: list[tuple[bytes, str]] = []
        for item in (getattr(outputs[-1], 'content', None) or []) if outputs else []:
            kind = getattr(item, 'type', None)
            if kind == 'text':
                texts.append(item.text or '')
                for annotation in getattr(item, 'annotations', None) or []:
                    url = getattr(annotation, 'url', None) or getattr(annotation, 'uri', None)
                    if url and url not in sources:
                        title = getattr(annotation, 'title', None) or url
                        sources[url] = {
                            'source': {'name': title, 'url': url},
                            'document': [title],
                            'metadata': [{'source': url, 'name': title}],
                        }
            elif kind == 'image' and getattr(item, 'data', None):
                images.append((base64.b64decode(item.data), getattr(item, 'mime_type', None) or 'image/png'))
        return ''.join(texts), list(sources.values()), images

    @staticmethod
    def _usage(interaction: Any) -> dict[str, Any] | None:
        """The interaction's usage in Open WebUI's shape (thought tokens count as output)."""
        if not (usage := interaction.usage):
            return None
        input_tokens = usage.total_input_tokens or 0
        output_tokens = usage.total_output_tokens or 0
        thought_tokens = usage.total_thought_tokens or 0
        total_tokens = usage.total_tokens or input_tokens + output_tokens + thought_tokens
        if total_tokens == input_tokens + output_tokens + thought_tokens:
            output_tokens += thought_tokens  # `total_output_tokens` does not include the thoughts.
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'prompt_tokens': input_tokens,
            'completion_tokens': output_tokens,
            'cache_read_input_tokens': usage.total_cached_tokens or 0,
            'completion_tokens_details': {'reasoning_tokens': thought_tokens},
        }


# endregion Interactions API models


# region Model table


class ModelCatalog:
    """The model table (an embedded copy of `gemini_models.yaml`) with the lookups the pipe needs."""

    def __init__(self, table: dict[str, Any]):
        self.table = table

    @classmethod
    def embedded(cls) -> ModelCatalog:
        return cls(yaml.safe_load(_MODEL_TABLE_YAML) or {})

    def entry(self, model_id: str) -> dict[str, Any]:
        entry = self.table.get(model_id)
        return entry if isinstance(entry, dict) else {}

    def capability(self, model_id: str, name: str, default: bool = False) -> bool:
        """A capability flag of the model; `default` applies to models that are not in the table."""
        capabilities = self.entry(model_id).get('capabilities')
        if not isinstance(capabilities, dict):
            return default
        return bool(capabilities.get(name, False))

    def is_image_model(self, model_id: str) -> bool:
        return self.capability(model_id, 'image_generation')

    def thinking(self, model_id: str) -> dict[str, Any]:
        """The model's `thinking` controls (levels or budget range); empty if it has none."""
        value = self.entry(model_id).get('thinking')
        return value if isinstance(value, dict) else {}

    def image_options(self, model_id: str) -> dict[str, Any]:
        """The model's `image_generation` options (sizes, aspect ratios, limits); empty for other models."""
        value = self.entry(model_id).get('image_generation')
        return value if isinstance(value, dict) else {}

    @staticmethod
    def generation(model_id: str) -> float:
        """The Gemini generation of a model id (3.0 for ids without a version)."""
        return Pipe._parse_gemini_version(model_id)


# The verbatim contents of plugins/pipes/gemini_models.yaml (tests/test_gemini_manifold_plus.py keeps them in sync).
_MODEL_TABLE_YAML: Final[str] = r"""
# Gemini Model Definitions
# Source for capabilities: https://aistudio.google.com/docs/models
# Source for thinking levels: https://aistudio.google.com/docs/thinking#levels-budgets
# Source for aspect ratios: https://aistudio.google.com/docs/image-generation#aspect_ratios_and_image_size
# Source for pricing: https://aistudio.google.com/docs/pricing
# Prices are per 1 million tokens in USD.
#
# Keys read by the pipe besides `capabilities`, `image_generation` and `pricing`:
#   thinking.levels / thinking.default: accepted `thinking_level` values (lowest to highest) and the API default.
#   thinking.budget / thinking.can_disable: accepted `thinking_budget` range of budget-only (Gemini 2.5) models,
#     and whether a budget of 0 turns thinking off.
#   image_generation.max_reference_images: hard limit on input images per request (omitted when undocumented).
#   image_generation.image_search: the model accepts Image Search grounding (`google_search.search_types`).
# Gemini Manifold Plus embeds a verbatim copy of this file (`_MODEL_TABLE_YAML`); keep both in sync
# (tests/test_gemini_manifold_plus.py checks it).


gemini-4-hydrogen:
  description: "Brings cutting-edge intelligence to everyone, with speed and efficiency for everyday coding and agentic workflows, featuring a unified 1M-token context window and long decode up to 1M tokens."
  # knowledge_cutoff: not published by Google.
  latest_update: "2026-11"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    # Context window is a unified input + output limit.
    input: 1048576
    # Default synchronous limit. Up to 1048576 with long decode (not used by this pipe).
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    # Implicit caching only. Explicit caching (cachedContents) is not supported.
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [low, medium, high, xhigh]
    default: high
  pricing:
    free_tier: true
    excluded_features: [search_grounding, grounding_google_maps]
    input:
      - up_to_tokens: null
        price_per_million: 1.50
    output:
      - up_to_tokens: null
        price_per_million: 7.50
    caching:
      - up_to_tokens: null
        price_per_million: 0.075

gemini-4-neon:
  description: "Brings frontier-level intelligence with fast end-to-end task completion across complex software engineering, knowledge work, and science workflows, featuring a unified 1M-token context window and long decode up to 1M tokens."
  # knowledge_cutoff: not published by Google.
  latest_update: "2026-11"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    # Context window is a unified input + output limit.
    input: 1048576
    # Default synchronous limit. Up to 1048576 with long decode (not used by this pipe).
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    # Implicit caching only. Explicit caching (cachedContents) is not supported.
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [low, medium, high, xhigh]
    default: high
  pricing:
    # Not available on the Free tier. Flat pricing across the full 1M context window.
    input:
      - up_to_tokens: null
        price_per_million: 3.00
    output:
      - up_to_tokens: null
        price_per_million: 15.00
    caching:
      - up_to_tokens: null
        price_per_million: 0.15

gemini-4-carbon:
  description: "Our most capable reasoning model, built for complex software engineering, mathematical and scientific problem solving, high factual accuracy, autonomous multi-step agents, and long decode up to 1M tokens."
  # knowledge_cutoff: not published by Google.
  latest_update: "2026-11"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    # Context window is a unified input + output limit.
    input: 1048576
    # Default synchronous limit. Up to 1048576 with long decode (not used by this pipe).
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    # Implicit caching only. Explicit caching (cachedContents) is not supported.
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [low, medium, high]
    default: medium
  pricing:
    # Not available on the Free tier.
    # Introductory pricing through 2026-11-30; standard pricing from 2026-12-01 is input 4.00, output 20.00, caching 0.20.
    input:
      - up_to_tokens: null
        price_per_million: 2.00
    output:
      - up_to_tokens: null
        price_per_million: 10.00
    caching:
      - up_to_tokens: null
        price_per_million: 0.10

gemini-4-argon:
  description: "Our flagship reasoning model, built for complex software engineering, deep mathematical and scientific problem solving, autonomous multi-step agents, and long-decode generation up to 1M tokens."
  # knowledge_cutoff: not published by Google.
  latest_update: "2026-10"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    # Context window is a unified input + output limit.
    input: 1048576
    # Default synchronous limit. Up to 1048576 with long decode (not used by this pipe).
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    # Implicit caching only. Explicit caching (cachedContents) is not supported.
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [low, medium, high]
    default: medium
  pricing:
    # Not available on the Free tier.
    # Introductory pricing through 2026-11-30; standard pricing from 2026-12-01 is input 4.00, output 20.00, caching 0.20.
    input:
      - up_to_tokens: null
        price_per_million: 2.00
    output:
      - up_to_tokens: null
        price_per_million: 10.00
    caching:
      - up_to_tokens: null
        price_per_million: 0.10

gemini-nano-banana-2.1:
  description: "Latest high-efficiency image generation and conversational editing model (Nano Banana 2.1). Flash-level speed and cost with improved visual quality, prompt adherence, multi-turn character consistency and text rendering. Supports Google Search grounding and 1K/2K/4K output."
  knowledge_cutoff: "2025-01"
  latest_update: "2026-10"
  supported_data_types:
    inputs: [Text, Image, Video, PDF]
    outputs: [Text, Image]
  tokens:
    input: 131072
    output: 32768
  capabilities:
    audio_generation: false
    batch_api: true
    caching: false
    code_execution: false
    file_search: false
    function_calling: false
    grounding_google_maps: false
    image_generation: true
    live_api: false
    search_grounding: true
    structured_outputs: false
    thinking: true
    url_context: false
  thinking:
    levels: [minimal, medium, high]
    default: medium
  image_generation:
    resolutions: ["1K", "2K", "4K"]
    aspect_ratios: ["1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"]
    max_reference_images: 14
    image_search: true
  pricing:
    input:
      - up_to_tokens: null
        price_per_million: 0.25
    output:
      - up_to_tokens: null
        price_per_million: 1.50
    image_output:
      - up_to_tokens: null
        price_per_million: 30.00

gemini-3.8-flash:
  description: "Gemini 3.8 Flash is our most intelligent Flash model, engineered for long-horizon software engineering, autonomous agents, and complex enterprise workflows—all with the speed and cost efficiency of Flash."
  knowledge_cutoff: "2026-03"
  latest_update: "2026-09"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    input: 1048576
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [low, medium, high]
    default: medium
  pricing:
    free_tier: true
    excluded_features: [search_grounding, grounding_google_maps]
    # Introductory pricing through 2026-12-31; standard pricing from 2027-01-01 is input 1.50, output 7.50, caching 0.15.
    input:
      - up_to_tokens: null
        price_per_million: 0.75
    output:
      - up_to_tokens: null
        price_per_million: 3.75
    caching:
      - up_to_tokens: null
        price_per_million: 0.075

gemini-3.1-flash-lite-image:
  description: "Our smallest and most cost effective image generation and editing model, built for at scale usage."
  knowledge_cutoff: "2025-01"
  latest_update: "2026-06"
  supported_data_types:
    inputs: [Text, Image]
    outputs: [Text, Image]
  tokens:
    input: 65536
    output: 4096
  capabilities:
    audio_generation: false
    batch_api: true
    caching: false
    code_execution: false
    file_search: false
    function_calling: false
    grounding_google_maps: false
    image_generation: true
    live_api: false
    search_grounding: false
    structured_outputs: false
    thinking: true
    url_context: false
  thinking:
    levels: [minimal, high]
    default: minimal
  image_generation:
    # Output is fixed at 1K; `image_size` is not configurable.
    resolutions: []
    aspect_ratios: ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
  pricing:
    input:
      - up_to_tokens: null
        price_per_million: 0.25
    output:
      - up_to_tokens: null
        price_per_million: 1.50
    image_output:
      - up_to_tokens: null
        price_per_million: 30.00

gemini-3.7-flash:
  # Deprecated on 2026-10-08; requests are routed to gemini-3.8-flash.
  description: "Gemini 3.7 Flash is the next iteration in the Gemini 3 series of highly-capable, natively multimodal, reasoning models."
  knowledge_cutoff: "2026-03"
  latest_update: "2026-08"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    input: 1048576
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [low, medium, high]
    default: medium
  pricing:
    free_tier: true
    excluded_features: [search_grounding, grounding_google_maps]
    # Introductory pricing through 2026-12-31; standard pricing from 2027-01-01 is input 1.50, output 7.50, caching 0.15.
    input:
      - up_to_tokens: null
        price_per_million: 0.75
    output:
      - up_to_tokens: null
        price_per_million: 3.75
    caching:
      - up_to_tokens: null
        price_per_million: 0.075

gemini-3.6-flash:
  description: "Our most token-efficient Flash model, combining frontier intelligence with strong agentic, coding, and search performance while using significantly fewer output tokens than Gemini 3.5 Flash, at a lower price."
  knowledge_cutoff: "2026-03"
  latest_update: "2026-07"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    input: 1048576
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [minimal, low, medium, high]
    default: medium
  pricing:
    free_tier: true
    excluded_features: [search_grounding, grounding_google_maps]
    # Introductory pricing through 2026-12-31; standard pricing from 2027-01-01 is input 1.50, output 7.50, caching 0.15.
    input:
      - up_to_tokens: null
        price_per_million: 0.75
    output:
      - up_to_tokens: null
        price_per_million: 3.75
    caching:
      - up_to_tokens: null
        price_per_million: 0.075

gemini-3.5-flash-lite:
  description: "A high-efficiency, low-latency model with upgraded agentic capabilities, optimized for cost-effective, high-throughput workloads."
  knowledge_cutoff: "2026-03"
  latest_update: "2026-07"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    input: 1048576
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [minimal, low, medium, high]
    default: minimal
  pricing:
    free_tier: true
    excluded_features: [search_grounding, grounding_google_maps]
    input:
      - up_to_tokens: null
        price_per_million: 0.30
    output:
      - up_to_tokens: null
        price_per_million: 2.50
    caching:
      - up_to_tokens: null
        price_per_million: 0.03

gemini-3.5-flash:
  # Deprecated on 2026-10-08; requests are routed to gemini-3.6-flash.
  description: "Gemini 3.5 Flash provides sustained frontier-level intelligence optimized for real-world tasks at a higher speed and lower cost. Designed for the agentic era, it excels at sub-agent deployment, multi-step workflows, and long-horizon tasks at scale. This model is particularly effective for rapid agentic loops involving complex coding cycles and iterations."
  knowledge_cutoff: "2025-01"
  latest_update: "2026-05"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    input: 1048576
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [minimal, low, medium, high]
    default: medium
  pricing:
    free_tier: true
    excluded_features: [search_grounding, grounding_google_maps]
    input:
      - up_to_tokens: null
        price_per_million: 1.50
    output:
      - up_to_tokens: null
        price_per_million: 9.00
    caching:
      - up_to_tokens: null
        price_per_million: 0.15

gemini-3.1-flash-lite:
  description: "Gemini 3.1 Flash-Lite is a low-latency, cost-effective multimodal model optimized for high-frequency, lightweight tasks. The model supports text, image, video, audio, and PDF inputs, and is designed for high-volume agentic workflows, simple data extraction, and applications where latency and API cost are the primary constraints."
  knowledge_cutoff: "2025-01"
  latest_update: "2026-05"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    input: 1048576
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [minimal, low, medium, high]
    default: minimal
  pricing:
    free_tier: true
    excluded_features: [search_grounding, grounding_google_maps]
    input:
      - up_to_tokens: null
        price_per_million: 0.25
    output:
      - up_to_tokens: null
        price_per_million: 1.50
    caching:
      - up_to_tokens: null
        price_per_million: 0.025

gemini-3.1-flash-image:
  # Deprecated on 2026-10-06 (no shutdown date); replacement: gemini-nano-banana-2.1.
  description: "Pro-level visual intelligence with Flash-speed efficiency and reality-grounded generation capabilities."
  knowledge_cutoff: "2025-01"
  latest_update: "2026-02"
  supported_data_types:
    inputs: [Text, Image, PDF]
    outputs: [Text, Image]
  tokens:
    input: 131072
    output: 32768
  capabilities:
    audio_generation: false
    batch_api: true
    caching: false
    code_execution: false
    file_search: false
    function_calling: false
    grounding_google_maps: false
    image_generation: true
    live_api: false
    search_grounding: true
    structured_outputs: false
    thinking: true
    url_context: false
  thinking:
    levels: [minimal, high]
    default: minimal
  image_generation:
    resolutions: ["512", "1K", "2K", "4K"]
    aspect_ratios: ["1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"]
    max_reference_images: 14
    image_search: true
  pricing:
    input:
      - up_to_tokens: null
        price_per_million: 0.25
    output:
      - up_to_tokens: null
        price_per_million: 1.50
    image_output:
      - up_to_tokens: null
        price_per_million: 60.00

gemini-3-flash-preview:
  description: "Our most intelligent model built for speed, combining frontier intelligence with superior search and grounding."
  knowledge_cutoff: "2025-01"
  latest_update: "2025-12"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    input: 1048576
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [minimal, low, medium, high]
    default: high
  pricing:
    free_tier: true
    excluded_features: [search_grounding, grounding_google_maps]
    input:
      - up_to_tokens: null
        price_per_million: 0.5
    output:
      - up_to_tokens: null
        price_per_million: 3
    caching:
      - up_to_tokens: null
        price_per_million: 0.05

gemini-3.1-pro-preview:
  description: "Refines the performance and reliability of the Gemini 3 Pro series with better thinking, improved token efficiency, and more grounded, factually consistent responses. Optimized for software engineering and agentic workflows requiring precise tool usage and reliable multi-step execution."
  knowledge_cutoff: "2025-01"
  latest_update: "2026-02"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    input: 1048576
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [low, medium, high]
    default: high
  pricing:
    input:
      - up_to_tokens: 200000
        price_per_million: 2.00
      - up_to_tokens: null
        price_per_million: 4.00
    output:
      - up_to_tokens: 200000
        price_per_million: 12.00
      - up_to_tokens: null
        price_per_million: 18.00
    caching:
      - up_to_tokens: 200000
        price_per_million: 0.20
      - up_to_tokens: null
        price_per_million: 0.40

gemini-3.1-pro-preview-customtools:
  description: "Custom tools variant of Gemini 3.1 Pro Preview, optimized for agentic workflows that use custom tools and bash. Better at prioritizing custom tools such as view_file or search_code."
  knowledge_cutoff: "2025-01"
  latest_update: "2026-02"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    input: 1048576
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    levels: [low, medium, high]
    default: high
  pricing:
    input:
      - up_to_tokens: 200000
        price_per_million: 2.00
      - up_to_tokens: null
        price_per_million: 4.00
    output:
      - up_to_tokens: 200000
        price_per_million: 12.00
      - up_to_tokens: null
        price_per_million: 18.00
    caching:
      - up_to_tokens: 200000
        price_per_million: 0.20
      - up_to_tokens: null
        price_per_million: 0.40

gemini-3-pro-image:
  description: "State-of-the-art image generation and editing model."
  knowledge_cutoff: "2025-01"
  latest_update: "2025-11"
  supported_data_types:
    inputs: [Text, Image]
    outputs: [Text, Image]
  tokens:
    input: 65536
    output: 32768
  capabilities:
    audio_generation: false
    batch_api: true
    caching: false
    code_execution: false
    file_search: false
    function_calling: false
    grounding_google_maps: false
    image_generation: true
    live_api: false
    search_grounding: true
    structured_outputs: false
    thinking: true
    url_context: false
  # Thinking is always on and has no level control.
  image_generation:
    resolutions: ["1K", "2K", "4K"]
    aspect_ratios: ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
    max_reference_images: 14
  pricing:
    input:
      - up_to_tokens: null
        price_per_million: 2.00
    output:
      - up_to_tokens: null
        price_per_million: 12.00
    image_output:
      - up_to_tokens: null
        price_per_million: 120.00

gemini-2.5-pro:
  # Deprecated; from 2026-12-01 requests are routed to gemini-3.8-flash and billed at its rates.
  description: "Our state-of-the-art thinking model, capable of reasoning over complex problems in code, math, and STEM."
  knowledge_cutoff: "2025-01"
  latest_update: "2025-11"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    input: 1048576
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    budget: {min: 128, max: 32768}
    can_disable: false
  pricing:
    free_tier: true
    excluded_features: [search_grounding, grounding_google_maps]
    input:
      - up_to_tokens: 200000
        price_per_million: 1.25
      - up_to_tokens: null
        price_per_million: 2.50
    output:
      - up_to_tokens: 200000
        price_per_million: 10.00
      - up_to_tokens: null
        price_per_million: 15.00
    caching:
      - up_to_tokens: 200000
        price_per_million: 0.125
      - up_to_tokens: null
        price_per_million: 0.25

gemini-2.5-flash:
  # Deprecated; from 2026-12-01 requests are routed to gemini-3.5-flash-lite and billed at its rates.
  description: "Our best model in terms of price-performance, offering well-rounded capabilities."
  knowledge_cutoff: "2025-01"
  latest_update: "2025-11"
  supported_data_types:
    inputs: [Text, Image, Video, Audio]
    outputs: [Text]
  tokens:
    input: 1048576
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    budget: {min: 0, max: 24576}
    can_disable: true
  pricing:
    free_tier: true
    input:
      - up_to_tokens: null
        price_per_million: 0.30
    output:
      - up_to_tokens: null
        price_per_million: 2.50
    caching:
      - up_to_tokens: null
        price_per_million: 0.03

gemini-2.5-flash-image:
  # Deprecated; shuts down on 2027-03-15. Replacement: gemini-3.1-flash-lite-image.
  description: "Our state-of-the-art image generation and editing model."
  knowledge_cutoff: "2025-06"
  latest_update: "2025-10"
  supported_data_types:
    inputs: [Text, Image]
    outputs: [Text, Image]
  tokens:
    input: 65536
    output: 32768
  capabilities:
    audio_generation: false
    batch_api: true
    caching: false
    code_execution: false
    file_search: false
    function_calling: false
    grounding_google_maps: false
    image_generation: true
    live_api: false
    search_grounding: false
    structured_outputs: false
    thinking: false
    url_context: false
  image_generation:
    resolutions: []
    aspect_ratios: ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
  pricing:
    input:
      - up_to_tokens: null
        price_per_million: 0.30
    output:
      - up_to_tokens: null
        price_per_million: 2.50
    image_output:
      - up_to_tokens: null
        price_per_million: 30.00

gemini-2.5-flash-lite:
  description: "Our fastest flash model optimized for cost-efficiency and high throughput."
  knowledge_cutoff: "2025-01"
  latest_update: "2025-11"
  supported_data_types:
    inputs: [Text, Image, Video, Audio, PDF]
    outputs: [Text]
  tokens:
    input: 1048576
    output: 65536
  capabilities:
    audio_generation: false
    batch_api: true
    caching: true
    code_execution: true
    file_search: true
    function_calling: true
    grounding_google_maps: true
    image_generation: false
    live_api: false
    search_grounding: true
    structured_outputs: true
    thinking: true
    url_context: true
  thinking:
    budget: {min: 512, max: 24576}
    can_disable: true
  pricing:
    free_tier: true
    input:
      - up_to_tokens: null
        price_per_million: 0.10
    output:
      - up_to_tokens: null
        price_per_million: 0.40
    caching:
      - up_to_tokens: null
        price_per_million: 0.01
"""  # noqa: E501

# endregion Model table
