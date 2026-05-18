import os

from datetime import datetime
from modules.utils import timer


class AtlasMemory:
    """
    MemPalace wrapper for Atlas persistent memory.
    Stores and retrieves conversations, preferences, and facts.
    """

    def __init__(self, config: dict):
        self.debug = True
        self.config = config.get("memory", {})
        self.enabled = self.config.get("enabled", False)
        self.wing = self.config.get("wing", "atlas")
        self.palace_path = os.path.expanduser(
            self.config.get("palace_path", "~/.mempalace/palace")
        )

        if self.enabled:
            self._init_mempalace()
            # self._init_embedder()  -> causing CUDA crash

    def _init_mempalace(self):
        """Initialize MemPalace searcher."""
        try:
            from mempalace.searcher import search_memories
            with timer("STT", self.debug):
                self._search_fn = search_memories
            print("[Memory] MemPalace initialized")
        except ImportError:
            print("[Memory] MemPalace not installed — pip install mempalace")
            self.enabled = False
        except Exception as e:
            print(f"[Memory] Init error: {e}")
            self.enabled = False

    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu"  # 🔥 FORCE CPU --- add to config.yaml
            )
            print("[Memory] Embedder loaded (CPU mode)")
        except Exception as e:
            print(f"[Memory] Embedder unavailable: {e}")
            self._embedder = None

    # ── Chunking ────────────────────────────────────────────────
    def chunk_text(self, text: str, max_chars: int = 800):
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    # ── Embedding (safe fallback) ───────────────────────────────
    def embed(self, text: str):
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            return embedder.encode(text).tolist()
        except Exception:
            return None  # fallback if model not available

    # ── Cosine similarity ───────────────────────────────────────
    def cosine_sim(self, a, b):
        import numpy as np
        if not a or not b:
            return 0.0
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ── Normalize memory  ────────────────────────────────────────────────
    def normalize_memory(self, mem):
        if isinstance(mem, dict):
            return {
                "content": mem.get("content") or mem.get("text"),
                "embedding": mem.get("embedding"),
                "agent": mem.get("agent"),
                "room": mem.get("room"),
                "wing": mem.get("wing"),
                "timestamp": mem.get("timestamp"),
                "access_count": mem.get("access_count", 0),
                "importance": mem.get("importance", 1.0),
            }

        if isinstance(mem, str):
            return {
                "content": mem,
                "embedding": None,
                "agent": None,
                "room": None,
                "wing": None,
                "timestamp": None,
                "access_count": 0,
                "importance": 1.0,
            }

        return None

    # ── Search ────────────────────────────────────────────────────────────

    def update_access(self, mem):
        mem["access_count"] = mem.get("access_count", 0) + 1

    def recall(self, query: str, n: int = 3):
        try:
            if not self._search_fn:
                return []

            print(f"[Memory] Searching for: {query}")

            results = self._search_fn(
                query=query,
                palace_path=self.palace_path
            )

            raw = results.get("results", [])

            # ── optional wing filter (manual) ─────────────
            raw = [
                r for r in raw
                if r.get("wing") in (self.wing, None)
            ]

            cleaned = []

            for m in raw[:n * 2]:  # oversample then trim
                text = m.get("text") or m.get("content")
                if not text:
                    continue

                text = text.strip()

                # hard filter out junk/code dumps
                if len(text) > 300:
                    continue

                cleaned.append(text[:200])

            print(f"[Memory] Returning {len(cleaned)} memories")

            return "\n".join(cleaned)


        except Exception as e:
            print(f"[Memory] Recall error: {e}")
            return ""

        # Advanced version to consider implementing later
        # try:
        #     from mempalace.miner import get_collection
        #
        #     collection = get_collection(self.palace_path)
        #     query_vec = self.embed(query)
        #
        #     results = []
        #
        #     for raw_mem in collection.get_all():
        #
        #         mem = self.normalize_memory(raw_mem)
        #         if not mem:
        #             continue
        #
        #         content = mem.get("content")
        #         if not content:
        #             continue
        #
        #         mem_vec = mem.get("embedding")
        #
        #         # ── Semantic similarity ─────────────────────
        #         score = self.cosine_sim(query_vec, mem_vec)
        #
        #         # ── Recency decay ───────────────────────────
        #         timestamp = mem.get("timestamp")
        #         if timestamp:
        #             try:
        #                 ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        #                 age_days = (datetime.now() - ts).days
        #             except:
        #                 age_days = 999
        #         else:
        #             age_days = 999
        #
        #         decay = np.exp(-0.02 * age_days)
        #
        #         # ── Importance + access boost ───────────────
        #         importance = mem.get("importance", 1.0)
        #         access = mem.get("access_count", 0)
        #         boost = np.log(access + 1) if access > 0 else 1.0
        #
        #         final_score = score * decay * importance * boost
        #
        #         results.append((final_score, mem))
        #
        #     results.sort(reverse=True, key=lambda x: x[0])
        #
        #     top_memories = [m for _, m in results[:n]]
        #
        #     return top_memories
        #
        # except Exception as e:
        #     print(f"[Memory] Recall error: {e}")
        #     return []

    # ── Store ─────────────────────────────────────────────────────────────

    def remember(self, text: str, room: str = "general") -> bool:
        """
        Store a memory explicitly.
        TEXT → CHUNK → TAG (agent/room/wing) → STORE
        """
        if not self.enabled:
            return False

        try:
            from mempalace.miner import add_drawer, get_collection

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            collection = get_collection(self.palace_path)

            chunks = self.chunk_text(text)

            # eventually want to add agents:
            # "atlas" → main AI
            # "system" → background processes (auto-summaries, indexing)
            # "user" → user-imported memories
            # "tool" → external integrations

            for i, chunk in enumerate(chunks):
                add_drawer(
                    collection=collection,
                    content=chunk,
                    source_file=f"memory_{timestamp}",
                    wing=self.wing,
                    room=room,
                    agent="atlas",
                    chunk_index=i
                )

            print(f"[Memory] Stored {len(chunks)} chunk(s)")
            return True

        except Exception as e:
            print(f"[Memory] Store error: {e}")
            return False

    def remember_conversation(self, command: str, response: str) -> bool:
        if not self.enabled:
            return False

        try:
            from mempalace.miner import add_drawer, get_collection

            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            content = f"[{timestamp_str}]\nUser: {command}\nAtlas: {response}"

            collection = get_collection(self.palace_path)
            chunks = self.chunk_text(content)

            for i, chunk in enumerate(chunks):
                add_drawer(
                    collection=collection,
                    content=chunk,
                    source_file=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    wing=self.wing,
                    room="conversations",
                    agent="atlas",
                    chunk_index=i
                )

            return True

        except Exception as e:
            print(f"[Memory] Conversation store error: {e}")
            return False

    # ── Wake-up context ───────────────────────────────────────────────────

    def get_wake_up_context(self) -> str | None:
        """Load critical facts on startup."""
        if not self.enabled:
            return None
        try:
            import subprocess
            result = subprocess.run(
                ['mempalace', 'wake-up', '--wing', self.wing],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                print("[Memory] Wake-up context loaded")
                return result.stdout.strip()
            return None
        except Exception as e:
            print(f"[Memory] Wake-up error: {e}")
            return None

    # ── Voice command helpers ─────────────────────────────────────────────

    def format_for_speech(self, memory_text: str) -> str:
        """Clean memory text for TTS."""
        import re
        if self.debug:
            print(f"[Observer]: formatting for speech: {memory_text}")
        text = re.sub(r'\[.*?\]:', '', memory_text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()[:500]
