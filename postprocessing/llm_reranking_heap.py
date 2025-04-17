import openai
import heapq
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()



# ---------------------------------------------------------------------------
#Typed containers
# ---------------------------------------------------------------------------
@dataclass
class BSTNode:
    """Node in the *preference* binary‑search tree.

    The invariant is:
        • All movies in **left** are preferred *over* ``movie_id``.
        • All movies in **right** are *less* preferred than ``movie_id``.

    Because every insertion uses the LLM to decide the preference between the
    incoming movie and the node's movie, the tree always respects this invariant.
    """

    movie_id: int
    left: Optional["BSTNode"] = None
    right: Optional["BSTNode"] = None

# ---------------------------------------------------------------------------
@dataclass
class LLMResult:
    preferred_movie: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# ---------------------------------------------------------------------------
class LLM_Heap_Reranker:

    DEFAULT_MODEL = "gpt-4o-mini"
    SYSTEM_INSTRUCTIONS = "You are a helpful movie expert that analyses user preferences and picks the best recommendation with concise reasoning."

    def __init__(
            self,
            users_df: pd.DataFrame,
            movies_df: pd.DataFrame,
            recs_df: pd.DataFrame,
            k: int = 20,
            model: str | None = None,
            api_key: str | None = None,
            max_llm_retries: int = 3,
    ) -> None:
        self.users_df = users_df.set_index("user_id")
        self.movies_df = movies_df.set_index("movie_id")
        self.recs_df = recs_df
        self.k = k
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_llm_retries = max_llm_retries

        # LLM client
        self.client = OpenAI(api_key=self.api_key)

        # Cache so we never ask the same question twice
        # Key = (user_id, min(movie1, movie2), max(movie1, movie2))
        self._comparison_cache: Dict[Tuple[int, int, int], int] = {}

        # Token accounting
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # -------------------------------------------------------------------------------------
    def rerank_all_users(self) -> pd.DataFrame:
        """Return a dataframe with the reranked list for every user."""

        results: List[pd.DataFrame] = []
        grouped = self.recs_df.groupby("user_id")

        for user_id, group in grouped:
            top_k_original = (
                group.sort_values("recommendation_rank", ascending=True)
                .head(self.k)["movie_id"]
                .tolist()
            )
            reranked_ids = self._rerank_for_user(user_id, top_k_original)
            results.append(
                pd.DataFrame(
                    {
                        "user_id": user_id,
                        "movie_id": reranked_ids,
                        "new_rank": list(range(1, len(reranked_ids) + 1)),
                    }
                )
            )
        combined = pd.concat(results, ignore_index=True)
        return combined

    # -------------------------------------------------------------------------------------
    def _get_user_profile(self, user_id: int) -> str:
        try:
            return self.users_df.at[user_id, "user_profile"]
        except KeyError:
            return "No user profile available."

    # -------------------------------------------------------------------------------------
    def _get_movie_info(self, movie_id: int) -> str:
        try:
            return self.movies_df.at[movie_id, "movie_info"]
        except KeyError:
            return "No movie information available."

    # -------------------------------------------------------------------------------------
    def _rerank_for_user(self, user_id: int, movie_ids: List[int]) -> List[int]:
        """Insert each movie into a preference‑BST and return the in‑order list."""
        if not movie_ids:
            return []

        # Build the tree root with the first movie id
        root = BSTNode(movie_id=movie_ids[0])

        for incoming_id in movie_ids[1:]:
            self._bst_insert(user_id, root, incoming_id)

        # In‑order traversal now gives a total order (best → worst)
        ordered: List[int] = []
        self._inorder(root, ordered)
        return ordered

    # -------------------------------------------------------------------------------------
    def _bst_insert(self, user_id: int, node: BSTNode, incoming_id: int) -> None:
        """Recursively insert *incoming_id* by LLM‑guided comparisons."""
        preferred = self._llm_prefers(user_id, incoming_id, node.movie_id)

        if preferred == incoming_id:
            # incoming is *better* → go left
            if node.left is None:
                node.left = BSTNode(movie_id=incoming_id)
            else:
                self._bst_insert(user_id, node.left, incoming_id)
        else:
            # incoming is *worse* → go right
            if node.right is None:
                node.right = BSTNode(movie_id=incoming_id)
            else:
                self._bst_insert(user_id, node.right, incoming_id)

    # -------------------------------------------------------------------------------------
    def _inorder(self, node: Optional[BSTNode], acc: List[int]) -> None:
        if node is None:
            return
        self._inorder(node.left, acc)
        acc.append(node.movie_id)
        self._inorder(node.right, acc)

    # -------------------------------------------------------------------------------------
    def _llm_prefers(self, user_id: int, a: int, b: int) -> int:
        """Return the *movie_id* preferred by the LLM (i.e. more relevant)."""
        # Ensure symmetric cache key
        key = (user_id, *sorted([a, b]))
        if key in self._comparison_cache:
            return self._comparison_cache[key]

        prompt = self._build_prompt(user_id, a, b)
        result = self._query_llm(prompt)
        preferred = int(result.preferred_movie)

        # Cache both (user,a,b) and (user,b,a) for symmetry
        self._comparison_cache[key] = preferred
        return preferred

    # ............................... LLM plumbing ..................................

    def _build_prompt(self, user_id: int, movie_id1: int, movie_id2: int) -> str:
        """Assemble the full user prompt shown to the LLM."""
        user_info = self._get_user_profile(user_id)
        movie_info_1 = self._get_movie_info(movie_id1)
        movie_info_2 = self._get_movie_info(movie_id2)

        prompt= f"""
Below is a user profile and everything we know about them:

{user_info}

Given these preferences, which of the two movies is **MORE relevant** to the user? Choose strictly *one*.

First option (movie_id = {movie_id1}):
{movie_info_1}

Second option (movie_id = {movie_id2}):
{movie_info_2}

Respond **only** with valid JSON following exactly this schema:
{{
  \"response\": \"{movie_id1} or {movie_id2}\"
}}
"""
        return prompt

    # -------------------------------------------------------------------------------------
    def _query_llm(self, prompt: str) -> LLMResult:
        """Low‑level OpenAI call with retry/backoff handling."""
        for attempt in range(self.max_llm_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=50,
                    temperature=0.7,
                )

                usage = response.usage
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens

                content = response.choices[0].message.content
                data = json.loads(content)
                preferred_movie = int(data["response"].strip())
                return LLMResult(
                    preferred_movie=preferred_movie,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )
            except json.JSONDecodeError:
                print("Invalid JSON from LLM (attempt %d)", attempt + 1)
            except Exception as exc:  # noqa: BLE001
                # APIConnectionError, RateLimitError, APIStatusError => wait & retry
                wait = 2 ** attempt
                time.sleep(wait)
        raise RuntimeError("Max LLM retry attempts exceeded.")

    # -------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
def loading_data(data, recommender_method):
    user_path = f"../Dataset/{data}/users_with_summary_df.csv"
    movie_path = f"movies/movies_structured_{data}.csv"
    recommendation_path = f"../output/{data}/{recommender_method}_recommendations_{data}.csv"

    users_df = pd.read_csv(user_path)
    users_df = users_df[['user_id', 'user_profile']]

    movies_df = pd.read_csv(movie_path)
    movies_df = movies_df[['movie_id', 'movie_info']]

    recommendation_df = pd.read_csv(recommendation_path)
    # module_source is a text representing the recommender system algorithm
    recommendation_df = recommendation_df[['user_id','movie_id', 'recommendation_rank', 'module_source']]

    return users_df, movies_df, recommendation_df



# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    data = "100k"
    method = "vgcf"
    users_df, movies_df, recs_df = loading_data(data=data, recommender_method="method")

    # --------- sample users --------------------
    # 1. pick 5 random users (set random_state for reproducibility)
    sample_users = (
        recs_df['user_id']
        .drop_duplicates()  # get unique users
        .sample(n=5, random_state=42)  # randomly pick 5 of them
    )

    # 2. filter your recommendations to those 5 users
    sampled_recs = recs_df[
        recs_df['user_id'].isin(sample_users)
    ]

    # --------- sample users --------------------


    reranker = LLM_Heap_Reranker(users_df, movies_df, recs_df, k=10)
    reranked_df = reranker.rerank_all_users()
    reranked_df.to_csv(f"{method}_recommendations_LLM_Heap.output", index=False)
    sampled_recs.to_csv(f"{method}_recommendations_LLM_Heap.output_sample", index=False)

    print("Saved reranked recommendations")

    # for every user in recommendation_df, select top k=20 recommendations
    # the goal is to rerank the recommendations using LLMs considering movie_info and user_profile

    # at every step, randomly chose 2 recommendations, and let the LLM decide which one is more relevant for the user
    # create a BST to preserve the ranking of recommendations (initially it has first two options)

    # every time, select another movie from the remaining recommendations and compare it with the root and continue the comparison until we find the correct spot for the recommendation.

    # the goal for having a BST is to minimize the total comparison that we need to do
    # for example, if movie X is more related than movie Y, in the next comparison, if the movie Z is more related than movie X, logically, it will be more related than movie Y as well and we don't need to do extra comparison