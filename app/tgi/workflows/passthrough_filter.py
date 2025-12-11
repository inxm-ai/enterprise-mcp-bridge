import logging
import re
import time
from difflib import SequenceMatcher
from typing import List, Optional, Sequence

from app.tgi.models import ChatCompletionRequest
from app.vars import TGI_MODEL_NAME


class PassThroughFilter:
    """
    Filters streaming pass-through updates to avoid spamming the user.

    Uses lightweight similarity heuristics first, then an optional LLM judge
    for borderline cases. Tracks per-agent cadence to avoid rapid-fire repeats.
    """

    def __init__(
        self,
        llm_client,
        model_name: str = TGI_MODEL_NAME,
        logger_obj: Optional[logging.Logger] = None,
        cooldown_seconds: float = 1.2,
        history_limit: int = 20,
    ) -> None:
        self.llm_client = llm_client
        self.model_name = model_name
        self.logger = logger_obj or logging.getLogger("uvicorn.error")
        self.cooldown_seconds = cooldown_seconds
        self.history_limit = history_limit
        self._last_emit_at: dict[str, float] = {}

    async def should_emit(
        self,
        candidate: str,
        agent_name: str,
        history: Sequence[str],
        user_message: str,
        pass_through_guideline: Optional[str],
        access_token: Optional[str],
        span,
    ) -> Optional[str]:
        """
        Decide whether a candidate pass-through update should be surfaced.

        Returns the (possibly trimmed) candidate when allowed, otherwise None.
        """
        clean_candidate = (candidate or "").strip()
        if not clean_candidate:
            return None

        normalized = self._normalize(clean_candidate)
        canonical = self._canonical(normalized)
        if not canonical:
            return None

        recent_history = list(history[-self.history_limit :]) if history else []
        history_keys = [self._canonical(self._normalize(h)) for h in recent_history]

        # Block exact canonical repeats
        if canonical in history_keys:
            return None

        # Numbers often signal progress (e.g., "2/10"); allow them to update even
        # when wording is similar.
        last_history = recent_history[-1] if recent_history else ""
        numbers_changed = self._numbers_changed(clean_candidate, last_history)
        progress_like = self._looks_like_progress(clean_candidate)

        borderline = False
        needs_judge = False
        for prior_key in reversed(history_keys):
            if not prior_key:
                continue
            seq_ratio = self._similarity(canonical, prior_key)
            if numbers_changed and progress_like:
                # Numeric progress deltas can be similar; allow them to pass through
                continue
            token_overlap = self._token_overlap(canonical, prior_key)
            containment = self._containment(canonical, prior_key)
            if seq_ratio >= 0.93:
                return None
            if seq_ratio >= 0.88 and token_overlap >= 0.6:
                return None
            if containment >= 0.9 and seq_ratio >= 0.84:
                return None
            if seq_ratio >= 0.78 or token_overlap >= 0.55:
                borderline = True

        if borderline and not (numbers_changed and progress_like):
            needs_judge = True
        if history_keys and numbers_changed and not progress_like:
            needs_judge = True

        # Judge rapid-fire bursts even if text is not obviously similar
        in_burst = self._is_within_window(agent_name, self.cooldown_seconds)
        if in_burst and not numbers_changed and history_keys:
            needs_judge = True
        elif borderline and not numbers_changed:
            needs_judge = True

        if needs_judge:
            judge_result = await self._judge_with_llm(
                candidate=clean_candidate,
                history=recent_history,
                agent_name=agent_name,
                user_message=user_message,
                pass_through_guideline=pass_through_guideline,
                access_token=access_token,
                span=span,
            )
            if judge_result is False:
                return None

        # Passed filters; record cadence and return normalized text with trailing break
        self._last_emit_at[agent_name] = time.monotonic()
        return (
            clean_candidate
            if clean_candidate.endswith("\n\n")
            else f"{clean_candidate}\n\n"
        )

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip().lower()

    def _canonical(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text or "").strip()

    def _token_overlap(self, a: str, b: str) -> float:
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return intersection / union if union else 0.0

    def _containment(self, a: str, b: str) -> float:
        """How much the smaller set is contained in the larger one."""
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        smaller, larger = (
            (tokens_a, tokens_b)
            if len(tokens_a) <= len(tokens_b)
            else (tokens_b, tokens_a)
        )
        if not smaller:
            return 0.0
        return len(smaller.intersection(larger)) / len(smaller)

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _numbers_changed(self, candidate: str, last: str) -> bool:
        nums_candidate = self._extract_numbers(candidate)
        nums_last = self._extract_numbers(last)
        if not nums_candidate or not nums_last:
            return False
        return nums_candidate != nums_last

    def _extract_numbers(self, text: str) -> List[str]:
        return re.findall(r"\d+", text or "")

    def _is_within_window(self, agent_name: str, window: float) -> bool:
        if window <= 0:
            return False
        last = self._last_emit_at.get(agent_name)
        if last is None:
            return False
        return (time.monotonic() - last) < window

    def _looks_like_progress(self, text: str) -> bool:
        """Detect simple progress-style messages (e.g., 2/10 or 3 out of 5)."""
        if not text:
            return False
        lowered = text.lower()
        if re.search(r"\b\d+\s*/\s*\d+\b", text):
            return True
        if re.search(r"\b\d+\s+(out of|of)\s+\d+\b", lowered):
            return True
        if re.search(r"\bprogress\b[:\s]*\d", lowered):
            return True
        return False

    async def _judge_with_llm(
        self,
        candidate: str,
        history: Sequence[str],
        agent_name: str,
        user_message: str,
        pass_through_guideline: Optional[str],
        access_token: Optional[str],
        span,
    ) -> Optional[bool]:
        """
        Ask a compact LLM judge whether the update is materially new.

        Returns True to allow, False to block, or None on failure (fallback allow).
        """
        if not self.llm_client or not hasattr(self.llm_client, "ask"):
            return None

        try:
            recent = history[-3:] if history else []
            system_prompt = (
                "You decide whether a streaming status update is worth showing.\n"
                "If the new update adds meaningful, user-visible information beyond the prior updates, respond ONLY with ALLOW.\n"
                "If it repeats or restates what was already shown, respond ONLY with BLOCK.\n"
                "Do not include explanations."
            )
            if pass_through_guideline:
                system_prompt += (
                    "\nFollow the style hint when you ALLOW: "
                    f"{pass_through_guideline}"
                )

            question = (
                f"User request: {user_message or '<none>'}\n"
                f"Agent: {agent_name}\n"
                f"Previous updates:\n- " + "\n- ".join(recent) + "\n\n"
                f"New update:\n{candidate}"
            )

            judge_request = ChatCompletionRequest(
                messages=[],
                model=self.model_name,
                stream=True,
                temperature=0,
                max_tokens=64,
                top_p=1,
            )

            result = await self.llm_client.ask(
                base_prompt=system_prompt,
                base_request=judge_request,
                question=question,
                access_token=access_token or "",
                outer_span=span,
            )
            decision = (result or "").strip().lower()
            if "block" in decision:
                return False
            if "allow" in decision:
                return True
            return None
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.debug(
                "[PassThroughFilter] LLM judge failed, allowing update: %s", exc
            )
            return None
