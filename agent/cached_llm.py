"""
CachedLLM wrapper for LiveKit integration
Provides semantic caching for LiveKit voice agents with observability
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncIterator, Union
import json
import os

import aiohttp
from livekit.agents import llm
from livekit.agents.llm import LLMStream, ChatMessage, ChatRole, ChatChunk, ChoiceDelta, ChatContext
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN

logger = logging.getLogger(__name__)

class CachedLLMStream(LLMStream):
    """Stream wrapper that handles both cached and live responses"""
    
    def __init__(self, 
                 cached_llm: 'CachedLLM',
                 chat_ctx: ChatContext,
                 tools: Optional[List[Any]] = None,
                 temperature: Optional[float] = None,
                 n: Optional[int] = None,
                 parallel_tool_calls: Optional[bool] = None,
                 tool_choice: Optional[Any] = None,
                 **kwargs):
        super().__init__(
            llm=cached_llm,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=DEFAULT_API_CONNECT_OPTIONS
        )
        self._cached_llm = cached_llm
        self._chat_ctx = chat_ctx
        self._tools = tools
        self._temperature = temperature
        self._n = n
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice
        self._kwargs = kwargs
        
        # State variables for caching after completion
        self._content_parts = []
    
    async def _run(self):
        """
        Core async producer method required by LLMStream.
        Handles cache lookup and response generation.
        """
        try:
            # Try cache lookup first
            cache_result = await self._cached_llm._lookup_cache(
                self._chat_ctx.items,
                model=getattr(self._cached_llm.fallback_llm, 'model', 'gpt-4o-mini'),
                temperature=self._temperature or 0.7,
                n=self._n,
                tool_choice=self._tool_choice,
                **self._kwargs
            )
            
            if cache_result and cache_result.get("hit"):
                # Cache hit - stream the cached response
                self._cached_llm._cache_stats["hits"] += 1
                cached_response = cache_result["response"]
                
                # Track latency savings
                estimated_llm_latency = 1500  # Assume 1.5s average LLM response
                actual_latency = cache_result.get("latency_ms", 50)
                latency_saved = max(0, estimated_llm_latency - actual_latency)
                self._cached_llm._cache_stats["total_latency_saved_ms"] += latency_saved
                
                logger.info(f"Cache hit! Streaming cached response")
                
                # Extract content and create chunk
                content = self._extract_content_from_response(cached_response)
                chunk = ChatChunk(
                    id="cached_response",
                    delta=ChoiceDelta(
                        content=content,
                        role="assistant"
                    )
                )
                
                # Push the chunk to the stream using the channel
                self._event_ch.send_nowait(chunk)
                
            else:
                # Cache miss - use fallback LLM and proxy the stream
                self._cached_llm._cache_stats["misses"] += 1
                logger.debug("Cache miss - using fallback LLM")
                
                # Prepare extra_kwargs with model-specific parameters
                extra_kwargs = dict(self._kwargs)
                if self._temperature is not None:
                    extra_kwargs['temperature'] = self._temperature
                if self._n is not None:
                    extra_kwargs['n'] = self._n
                
                live_stream = self._cached_llm.fallback_llm.chat(
                    chat_ctx=self._chat_ctx,
                    tools=self._tools,
                    parallel_tool_calls=self._parallel_tool_calls,
                    tool_choice=self._tool_choice,
                    extra_kwargs=extra_kwargs if extra_kwargs else NOT_GIVEN
                )
                
                # Proxy all chunks from the live stream
                collected_content = []
                async with live_stream:
                    async for chunk in live_stream:
                        # Forward the chunk to our stream using the channel
                        self._event_ch.send_nowait(chunk)
                        
                        # Collect content for caching
                        if chunk.delta and chunk.delta.content:
                            collected_content.append(chunk.delta.content)
                
                # Cache the complete response
                if collected_content:
                    full_response = "".join(collected_content)
                    response_data = self._cached_llm._convert_response_to_cache_format(full_response)
                    await self._cached_llm._store_cache(
                        self._chat_ctx.items,
                        response_data,
                        model=getattr(self._cached_llm.fallback_llm, 'model', 'gpt-4o-mini'),
                        temperature=self._temperature or 0.7,
                        tool_choice=self._tool_choice,
                        **self._kwargs
                    )
                    logger.debug("Response cached successfully")
                
        except Exception as e:
            logger.error(f"Error in CachedLLMStream._run: {e}")
            # Re-raise the exception to let the base class handle it
            raise
    
    def _extract_content_from_response(self, response: Dict[str, Any]) -> str:
        """Extract content from cached response"""
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        
        # Fallback content extraction
        if "content" in response:
            return response["content"]
        
        return str(response)
    
    def get_collected_content(self) -> str:
        """Get all collected content from the stream"""
        return "".join(self._content_parts)
    
    async def _cache_response_after_completion(self):
        """Cache the response after stream completion"""
        try:
            # Wait a bit for stream to complete
            await asyncio.sleep(0.1)
            
            # Get the collected content
            content = self.get_collected_content()
            
            if content:
                # Create response in cache format
                response_data = self._cached_llm._convert_response_to_cache_format(content)
                
                # Store in cache
                await self._cached_llm._store_cache(
                    self._chat_ctx.items, 
                    response_data,
                    model=getattr(self._cached_llm.fallback_llm, 'model', 'gpt-4o-mini'),
                    temperature=self._temperature or 0.7,
                    tool_choice=self._tool_choice,
                    **self._kwargs
                )
                
                logger.debug(f"LLM response cached successfully")
            
        except Exception as e:
            logger.error(f"Error caching response after completion: {e}")

class CachedLLM(llm.LLM):
    """
    LiveKit LLM wrapper with semantic caching
    Provides context-aware caching for voice agent interactions
    """
    
    def __init__(self,
                 fallback_llm: llm.LLM,
                 cache_endpoint: str = "http://localhost:8000",
                 enable_caching: bool = True,
                 similarity_threshold: float = 0.85,
                 cache_timeout: float = 5.0):
        super().__init__()
        
        self.fallback_llm = fallback_llm
        self.cache_endpoint = cache_endpoint
        self.enable_caching = enable_caching
        self.similarity_threshold = similarity_threshold
        self.cache_timeout = cache_timeout
        
        # HTTP session for cache communication
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_timeout = aiohttp.ClientTimeout(total=cache_timeout)
        
        # Circuit breaker for cache failures
        self._cache_failures = 0
        self._max_cache_failures = 5
        self._failure_reset_time = 300  # 5 minutes
        self._last_failure_time = 0
        
        # Metrics integration
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "total_latency_saved_ms": 0
        }
    
    async def _ensure_session(self):
        """Ensure HTTP session is available"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._session_timeout)
    
    async def _close_session(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _is_cache_available(self) -> bool:
        """Check if cache is available (circuit breaker logic)"""
        if not self.enable_caching:
            return False
        
        current_time = time.time()
        
        # Reset failure count after timeout
        if current_time - self._last_failure_time > self._failure_reset_time:
            self._cache_failures = 0
        
        return self._cache_failures < self._max_cache_failures
    
    def _record_cache_failure(self):
        """Record cache failure for circuit breaker"""
        self._cache_failures += 1
        self._last_failure_time = time.time()
        self._cache_stats["errors"] += 1
        
        if self._cache_failures >= self._max_cache_failures:
            logger.warning(f"Cache circuit breaker activated after {self._cache_failures} failures")
    
    async def _lookup_cache(self, messages: List[Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Look up cached response"""
        if not self._is_cache_available():
            return None
        
        try:
            await self._ensure_session()
            
            # Convert LiveKit messages to cache format
            cache_messages = self._convert_messages_to_cache_format(messages)
            
            payload = {
                "messages": cache_messages,
                "model": kwargs.get("model", "gpt-4o-mini"),
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens")
            }
            
            start_time = time.time()
            
            async with self._session.post(
                f"{self.cache_endpoint}/cache/lookup",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    lookup_time = (time.time() - start_time) * 1000
                    
                    if result.get("hit"):
                        self._cache_stats["hits"] += 1
                        logger.info(f"Cache hit! Similarity: {result.get('similarity_score', 0):.3f}, "
                                  f"Lookup time: {lookup_time:.2f}ms")
                        return result
                    else:
                        self._cache_stats["misses"] += 1
                        logger.debug(f"Cache miss, lookup time: {lookup_time:.2f}ms")
                        return None
                else:
                    logger.warning(f"Cache lookup failed with status {response.status}")
                    self._record_cache_failure()
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning("Cache lookup timeout")
            self._record_cache_failure()
            return None
        except Exception as e:
            logger.error(f"Cache lookup error: {e}")
            self._record_cache_failure()
            return None
    
    async def _store_cache(self, messages: List[Any], response: Dict[str, Any], **kwargs):
        """Store response in cache"""
        if not self._is_cache_available():
            return
        
        try:
            await self._ensure_session()
            
            # Convert messages and response to cache format
            cache_messages = self._convert_messages_to_cache_format(messages)
            
            payload = {
                "messages": cache_messages,
                "response": response,
                "model": kwargs.get("model", "gpt-4o-mini"),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            async with self._session.post(
                f"{self.cache_endpoint}/cache/store",
                json=payload
            ) as store_response:
                if store_response.status == 200:
                    result = await store_response.json()
                    logger.debug(f"Response cached successfully in {result.get('latency_ms', 0):.2f}ms")
                else:
                    logger.warning(f"Cache store failed with status {store_response.status}")
                    
        except Exception as e:
            logger.error(f"Cache store error: {e}")
    
    def _convert_messages_to_cache_format(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Convert LiveKit ChatItems to cache format"""
        cache_messages = []
        
        for item in messages:
            # Only process ChatMessage items, skip function calls etc
            if hasattr(item, 'type') and item.type == 'message':
                # Extract text content from the content list
                text_content = ""
                if hasattr(item, 'content') and item.content:
                    # ChatMessage.content is a list of ChatContent (str, ImageContent, AudioContent)
                    text_parts = [c for c in item.content if isinstance(c, str)]
                    text_content = "\n".join(text_parts) if text_parts else ""
                
                cache_msg = {
                    "role": item.role.value if hasattr(item.role, 'value') else str(item.role),
                    "content": text_content
                }
                cache_messages.append(cache_msg)
        
        return cache_messages
    
    def _convert_response_to_cache_format(self, response_content: str) -> Dict[str, Any]:
        """Convert response content to cache format"""
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "index": 0
            }],
            "usage": {
                "cached": True
            }
        }
    
    def chat(self,
             *,
             chat_ctx: ChatContext,
             tools: Optional[List[Any]] = None,
             conn_options: Any = DEFAULT_API_CONNECT_OPTIONS,
             parallel_tool_calls: Any = NOT_GIVEN,
             tool_choice: Any = NOT_GIVEN,
             extra_kwargs: Any = NOT_GIVEN) -> "LLMStream":
        """
        Main chat method with caching integration
        Returns a CachedLLMStream that handles async operations in __aenter__
        """
        # Extract temperature and other model-specific parameters from extra_kwargs
        temperature = None
        n = None
        kwargs = {}
        
        if extra_kwargs is not NOT_GIVEN:
            kwargs = dict(extra_kwargs)
            temperature = kwargs.pop('temperature', None)
            n = kwargs.pop('n', None)
        
        # Return stream object immediately (synchronously)
        # All async work will happen in the stream's __aenter__ method
        return CachedLLMStream(
            cached_llm=self,
            chat_ctx=chat_ctx,
            tools=tools,
            temperature=temperature,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            **kwargs
        )
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / max(1, total_requests)
        
        return {
            "cache_enabled": self.enable_caching,
            "cache_available": self._is_cache_available(),
            "cache_endpoint": self.cache_endpoint,
            "hit_rate": round(hit_rate, 4),
            "total_requests": total_requests,
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "cache_errors": self._cache_stats["errors"],
            "total_latency_saved_ms": self._cache_stats["total_latency_saved_ms"],
            "cache_failures": self._cache_failures,
            "circuit_breaker_active": self._cache_failures >= self._max_cache_failures
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of cache and fallback LLM"""
        health_status = {
            "cache_healthy": False,
            "fallback_llm_healthy": True,  # Assume healthy unless we can check
            "cache_endpoint": self.cache_endpoint
        }
        
        try:
            await self._ensure_session()
            async with self._session.get(f"{self.cache_endpoint}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    health_status["cache_healthy"] = health_data.get("status") == "healthy"
                    health_status["cache_details"] = health_data
                    
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            health_status["cache_error"] = str(e)
        
        return health_status
    
    async def aclose(self):
        """Close the cached LLM and clean up resources"""
        await self._close_session()
        
        if hasattr(self.fallback_llm, 'aclose'):
            await self.fallback_llm.aclose()
        
        logger.info("CachedLLM closed successfully")