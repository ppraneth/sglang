from __future__ import annotations

"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Public re-exports for the allocator package.

Classes that have been promoted to their own sub-module are imported here so
that all existing ``from sglang.srt.mem_cache.allocator import X`` call-sites
continue to work without modification.

"""

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import (
    PagedTokenToKVPoolAllocator,
    alloc_decode_kernel,
    alloc_extend_kernel,
    alloc_extend_naive,
)
from sglang.srt.mem_cache.allocator.standard import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

__all__ = [
    "BaseTokenToKVPoolAllocator",
    "TokenToKVPoolAllocator",
    "PagedTokenToKVPoolAllocator",
    "alloc_extend_naive",
    "alloc_extend_kernel",
    "alloc_decode_kernel",
    "HiSparseTokenToKVPoolAllocator",
    "SWATokenToKVPoolAllocator",
]
