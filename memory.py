"""
MemoryManager: four-layer memory architecture for Nova.

  working    — RAM deque, current session only, destroyed on clear_working()
  episodic   — ChromaDB vector store (if available), else in-memory list
  semantic   — JSON file, structured user preferences (e.g. preferred AC temp)
  procedural — JSON file, successful trigger→action patterns learned over time
"""

import json
import uuid
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import chromadb
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False

from config import (
    MEMORY_DIR,
    WORKING_MAXLEN,
    SKILL_SIM_THRESHOLD,
    EPISODE_DIST_CUTOFF,
)


class _InMemoryEpisodic:
    """Fallback episodic store when chromadb is not available."""

    def __init__(self):
        self._entries: List[Dict] = []

    def count(self) -> int:
        return len(self._entries)

    def add(self, ids, embeddings, documents, metadatas):
        for eid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self._entries.append({
                "id": eid,
                "embedding": np.array(emb, dtype=np.float32),
                "document": doc,
                "metadata": meta,
            })

    def query(self, query_embeddings, n_results, include=None):
        if not self._entries:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        q = np.array(query_embeddings[0], dtype=np.float32)
        q_norm = np.linalg.norm(q) + 1e-9
        scored = []
        for e in self._entries:
            v = e["embedding"]
            cos_dist = 1.0 - float(np.dot(q, v) / (q_norm * (np.linalg.norm(v) + 1e-9)))
            scored.append((cos_dist, e))
        scored.sort(key=lambda x: x[0])
        top = scored[:n_results]
        return {
            "documents": [[t[1]["document"] for t in top]],
            "metadatas": [[t[1]["metadata"] for t in top]],
            "distances": [[t[0] for t in top]],
        }


class MemoryManager:

    def __init__(
        self,
        embed_fn: Callable[[str], List[float]],
        persist_dir: str = MEMORY_DIR,
        working_maxlen: int = WORKING_MAXLEN,
    ):
        self._embed = embed_fn
        Path(persist_dir).mkdir(exist_ok=True)

        # Working memory — session RAM only
        self.working: deque = deque(maxlen=working_maxlen)

        # Episodic memory — ChromaDB or in-memory fallback
        if _HAS_CHROMADB:
            self._chroma = chromadb.PersistentClient(path=f"{persist_dir}/chroma")
            self.episodes = self._chroma.get_or_create_collection(
                name="episodes", metadata={"hnsw:space": "cosine"}
            )
        else:
            self.episodes = _InMemoryEpisodic()

        # Semantic memory — JSON
        self._prefs_path = Path(f"{persist_dir}/user_prefs.json")
        self.prefs: Dict[str, Any] = (
            json.loads(self._prefs_path.read_text())
            if self._prefs_path.exists()
            else {}
        )

        # Procedural memory — JSON
        self._skills_path = Path(f"{persist_dir}/skills.json")
        self.skills: List[Dict] = (
            json.loads(self._skills_path.read_text())
            if self._skills_path.exists()
            else []
        )

    # ── Working memory ────────────────────────────────────────────────────────

    def push_working(self, role: str, text: str):
        self.working.append({
            "role": role,
            "text": text,
            "ts": datetime.now().isoformat(timespec="seconds"),
        })

    def clear_working(self):
        self.working.clear()

    def working_as_text(self) -> str:
        return "\n".join(f"{t['role'].upper()}: {t['text']}" for t in self.working)

    # ── Semantic memory ───────────────────────────────────────────────────────

    def update_pref(self, key: str, value: Any):
        self.prefs[key] = value
        self._prefs_path.write_text(json.dumps(self.prefs, indent=2, ensure_ascii=False))

    # ── Procedural memory ─────────────────────────────────────────────────────

    def _save_skills(self):
        self._skills_path.write_text(json.dumps(self.skills, indent=2, ensure_ascii=False))

    def record_skill(self, trigger: str, action: Dict):
        """Increment count if seen before; otherwise add new skill."""
        for s in self.skills:
            if s["trigger"] == trigger and s["action"] == action:
                s["count"] += 1
                s["last_used"] = datetime.now().isoformat(timespec="seconds")
                self._save_skills()
                return
        self.skills.append({
            "trigger":   trigger,
            "action":    action,
            "count":     1,
            "last_used": datetime.now().isoformat(timespec="seconds"),
        })
        self._save_skills()

    def lookup_skill(self, trigger_text: str) -> Optional[Dict]:
        """Return the most similar known skill if cosine similarity > threshold."""
        if not self.skills:
            return None
        q = np.array(self._embed(trigger_text))
        best, best_sim = None, -1.0
        for s in self.skills:
            sv = np.array(self._embed(s["trigger"]))
            sim = float(np.dot(q, sv) / (np.linalg.norm(q) * np.linalg.norm(sv) + 1e-9))
            if sim > best_sim:
                best_sim, best = sim, s
        return best if best_sim > SKILL_SIM_THRESHOLD else None

    # ── Episodic memory ───────────────────────────────────────────────────────

    def save_episode(self, user_text: str, result_type: str, nova_reply: str = ""):
        self.episodes.add(
            ids=[str(uuid.uuid4())],
            embeddings=[self._embed(user_text)],
            documents=[user_text],
            metadatas=[{
                "ts":          datetime.now().isoformat(timespec="seconds"),
                "result_type": result_type,
                "nova_reply":  nova_reply[:300],
            }],
        )

    def retrieve_episodes(self, query: str, n: int = 3) -> List[Dict]:
        count = self.episodes.count()
        if count == 0:
            return []
        results = self.episodes.query(
            query_embeddings=[self._embed(query)],
            n_results=min(n, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": doc, "meta": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # ── Context builder — RAG prompt prefix ──────────────────────────────────

    def build_context(self, current_input: str, max_episodes: int = 3) -> str:
        """
        Aggregate all memory layers into a context block for the LLM.

        Ledger   : save_episode() is the append-only raw log.
        Derived  : this method produces the view the LLM actually sees.
        Temporal : cosine distance acts as a recency proxy (recent ≈ relevant).
        """
        sections = []

        # Episodic — semantically similar past interactions
        eps = self.retrieve_episodes(current_input, n=max_episodes)
        relevant = [
            f"[{ep['meta']['ts'][:16]}] User: {ep['text']} → Nova: {ep['meta']['nova_reply']}"
            for ep in eps
            if ep["distance"] < EPISODE_DIST_CUTOFF
        ]
        if relevant:
            sections.append("## Relevant past interactions\n" + "\n".join(relevant))

        # Semantic — user preferences
        if self.prefs:
            lines = [f"- {k}: {v}" for k, v in self.prefs.items()]
            sections.append("## User preferences\n" + "\n".join(lines))

        # Procedural — top learned patterns
        if self.skills:
            top = sorted(self.skills, key=lambda x: x["count"], reverse=True)[:3]
            lines = [f'- "{s["trigger"]}" → {s["action"]} (used {s["count"]}x)' for s in top]
            sections.append("## Learned user patterns\n" + "\n".join(lines))

        # Working — current session window
        wm = self.working_as_text()
        if wm:
            sections.append("## Current session\n" + wm)

        return "\n\n".join(sections)

    # ── Debug summary ─────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        return {
            "working_turns":  len(self.working),
            "episodic_count": self.episodes.count(),
            "prefs":          self.prefs,
            "skills_count":   len(self.skills),
        }
