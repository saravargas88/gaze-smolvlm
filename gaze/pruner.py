"""
gaze/pruner.py

The core idea of this whole project in one file:
  SmolVLM processes an image by chopping it into a grid of patches.
  Each patch becomes a "visual token" that the language model reads.
  The more tokens, the slower decoding is.

  We use gaze — where the user is actually looking — to figure out
  which patches actually matter for the question being asked, and
  throw away the ones that don't.

  Two modes:
    1. Gaze-guided  : keep patches near where the user looked
    2. Attention     : keep patches the model paid the most attention to
                       (no gaze needed — useful as a baseline)
"""

import math

import torch
import torch.nn.functional as F


class GazePruner:
    """
    Takes a grid of visual tokens and a gaze point, and returns a
    smaller set of tokens — the ones worth keeping.

    How to think about this:
      Imagine the image is a 14x14 grid of tiles (196 tiles total for SmolVLM-256M).
      The user's gaze lands somewhere in that grid.
      We keep the tiles closest to the gaze point and drop the rest.
      Fewer tiles = fewer tokens = faster decoding.

    Args:
        grid_h: number of patch rows the image was split into
        grid_w: number of patch columns the image was split into
        keep_ratio: fraction of tokens to keep (0.5 = keep half)
        min_tokens: never drop below this many tokens, no matter what
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        keep_ratio: float = 0.5,
        min_tokens: int = 16,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens

        # Pre-compute where each token lives in the image.
        # Token i is at row (i // grid_w), col (i % grid_w).
        # We normalize to [0, 1] so gaze coordinates can be in the same space.
        n_tokens = grid_h * grid_w
        rows = torch.arange(n_tokens) // grid_w          # which row is token i in?
        cols = torch.arange(n_tokens) % grid_w           # which col is token i in?
        self.token_row_norm = rows.float() / grid_h      # normalized row position [0, 1]
        self.token_col_norm = cols.float() / grid_w      # normalized col position [0, 1]

    def gaze_scores(self, gaze_x: float, gaze_y: float) -> torch.Tensor:
        """
        Score each token by how close it is to the gaze point.
        Closer = higher score = more likely to be kept.

        gaze_x, gaze_y: gaze coordinates, both in [0, 1].
                        (0,0) is top-left of the image,
                        (1,1) is bottom-right.

        Returns a tensor of shape (n_tokens,) with scores between 0 and 1.
        Higher score = closer to gaze = more important.
        """
        # Euclidean distance from each token center to the gaze point.
        # This is just the straight-line distance in normalized image coordinates.
        dist = torch.sqrt(
            (self.token_col_norm - gaze_x) ** 2
            + (self.token_row_norm - gaze_y) ** 2
        )

        # Flip so that close tokens get HIGH scores (we want top-k, not bottom-k)
        # Max possible distance on a unit square is sqrt(2) ≈ 1.41
        max_dist = math.sqrt(2)
        score = 1.0 - (dist / max_dist)

        return score  # shape: (n_tokens,)

    def gaussian_gaze_scores(
        self, gaze_x: float, gaze_y: float, sigma: float = 0.2
    ) -> torch.Tensor:
        """
        Same idea as gaze_scores(), but softer: uses a Gaussian falloff
        instead of a hard distance. Tokens very close to gaze get ~1.0,
        tokens far away get close to 0. sigma controls the spread.

        A smaller sigma = tighter focus around the gaze point.
        A larger sigma = more tokens get kept (more spread out weighting).

        This is the version we actually use in experiments because it's
        smoother and doesn't have a sharp cutoff edge.
        """
        dist_sq = (
            (self.token_col_norm - gaze_x) ** 2
            + (self.token_row_norm - gaze_y) ** 2
        )
        score = torch.exp(-dist_sq / (2 * sigma ** 2))
        return score  # shape: (n_tokens,)

    def prune_by_gaze(
        self,
        visual_tokens: torch.Tensor,
        gaze_x: float,
        gaze_y: float,
        sigma: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Main pruning function. Given visual tokens and a gaze point,
        return only the tokens worth keeping.

        Args:
            visual_tokens: shape (n_tokens, hidden_dim) — the full set of
                           patch embeddings from the vision encoder.
            gaze_x: horizontal gaze position in [0, 1], 0 = left edge
            gaze_y: vertical gaze position in [0, 1], 0 = top edge
            sigma: Gaussian spread (see gaussian_gaze_scores)

        Returns:
            kept_tokens: shape (n_kept, hidden_dim) — the pruned token set
            keep_idx:    shape (n_kept,) — original indices of kept tokens
                         (useful for debugging / visualizing which patches survived)
        """
        n_tokens = visual_tokens.shape[0]
        n_keep = max(self.min_tokens, int(n_tokens * self.keep_ratio))

        # Score every token by proximity to gaze
        scores = self.gaussian_gaze_scores(gaze_x, gaze_y, sigma)

        # Pick the top-n_keep tokens by score
        # topk returns values and indices; we only need the indices
        _, keep_idx = torch.topk(scores, k=n_keep)

        # Sort the indices so token order is preserved (models care about order)
        keep_idx, _ = torch.sort(keep_idx)

        kept_tokens = visual_tokens[keep_idx]

        return kept_tokens, keep_idx

    def prune_by_attention(
        self,
        visual_tokens: torch.Tensor,
        attention_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Baseline pruning using attention weights instead of gaze.
        This is the SparseVLM-style approach: keep whichever tokens the
        model was already paying attention to, drop the rest.

        No gaze data needed — this is useful when you want to compare
        "gaze-guided" vs "attention-guided" to see if gaze actually helps.

        Args:
            visual_tokens:    shape (n_tokens, hidden_dim)
            attention_scores: shape (n_tokens,) — how much the model attends
                              to each visual token (e.g. mean attention weight
                              across heads, averaged over the first few decode steps)

        Returns:
            kept_tokens, keep_idx  (same as prune_by_gaze)
        """
        n_tokens = visual_tokens.shape[0]
        n_keep = max(self.min_tokens, int(n_tokens * self.keep_ratio))

        _, keep_idx = torch.topk(attention_scores, k=n_keep)
        keep_idx, _ = torch.sort(keep_idx)

        kept_tokens = visual_tokens[keep_idx]

        return kept_tokens, keep_idx

    def prune_combined(
        self,
        visual_tokens: torch.Tensor,
        gaze_x: float,
        gaze_y: float,
        attention_scores: torch.Tensor,
        gaze_weight: float = 0.6,
        sigma: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Blend gaze and attention into one score.

        The idea: gaze tells us where the *user* was looking, attention
        tells us what the *model* found important. Combining them should
        be better than either alone.

        gaze_weight controls the blend:
          1.0 = pure gaze,  0.0 = pure attention,  0.6 = gaze-leaning blend
        """
        gaze_s = self.gaussian_gaze_scores(gaze_x, gaze_y, sigma)

        # Normalize attention to [0, 1] so it's on the same scale as gaze scores
        attn_s = attention_scores.float()
        attn_s = (attn_s - attn_s.min()) / (attn_s.max() - attn_s.min() + 1e-8)

        combined = gaze_weight * gaze_s + (1.0 - gaze_weight) * attn_s

        n_tokens = visual_tokens.shape[0]
        n_keep = max(self.min_tokens, int(n_tokens * self.keep_ratio))

        _, keep_idx = torch.topk(combined, k=n_keep)
        keep_idx, _ = torch.sort(keep_idx)

        return visual_tokens[keep_idx], keep_idx


class TilePruner:
    """
    Tile-level pruning for SmolVLM / Idefics3.

    SmolVLM doesn't just encode the image once — it splits it into tiles
    first, encodes each tile separately, then concatenates all the tokens:

        tile 0          : 1 global tile  — low-res view of the whole image
        tiles 1-16      : 4x4 grid of local tiles — high-res patches

    Each tile produces 64 tokens → total = 17 × 64 = 1,088 visual tokens.

    The insight: if the user is looking at the top-left of the image, we
    probably don't need the 9 tiles covering the bottom-right. Dropping
    those entire tiles removes 9 × 64 = 576 tokens from the sequence,
    which directly reduces decoder compute.

    We always keep tile 0 (the global view) because it gives the model
    a low-res "overview" of the whole scene — dropping it hurts quality.

    Args:
        n_local_tiles_side: the grid is n x n, default 4 (so 4x4=16 tiles)
        keep_ratio: fraction of LOCAL tiles to keep (tile 0 always kept)
        min_local_tiles: never drop below this many local tiles
    """

    TOKENS_PER_TILE = 64  # fixed by SmolVLM architecture (8x8 after pixel shuffle)

    def __init__(
        self,
        n_local_tiles_side: int = 4,
        keep_ratio: float = 0.5,
        min_local_tiles: int = 2,
    ):
        self.n_local = n_local_tiles_side  # 4 → 4x4 = 16 local tiles
        self.keep_ratio = keep_ratio
        self.min_local_tiles = min_local_tiles

    def tile_scores(self, gaze_x: float, gaze_y: float, sigma: float = 0.3) -> torch.Tensor:
        """
        Score each of the 16 local tiles by how close its center is to the gaze point.

        Tiles are laid out in a 4x4 grid. Tile centers are at:
          col = 0, 1, 2, 3  →  normalized x = 0/3, 1/3, 2/3, 3/3
          row = 0, 1, 2, 3  →  normalized y = 0/3, 1/3, 2/3, 3/3

        Returns shape (16,) with scores in [0, 1].
        """
        n = self.n_local
        scores = []
        for row in range(n):
            for col in range(n):
                cx = col / (n - 1)   # tile center x, normalized to [0, 1]
                cy = row / (n - 1)   # tile center y, normalized to [0, 1]
                dist_sq = (cx - gaze_x) ** 2 + (cy - gaze_y) ** 2
                score = math.exp(-dist_sq / (2 * sigma ** 2))
                scores.append(score)
        return torch.tensor(scores)

    def select_tiles(self, gaze_x: float, gaze_y: float, sigma: float = 0.3) -> list[int]:
        """
        Return the indices (into image_hidden_states) of tiles to KEEP.

        Index 0  = global tile  → always kept
        Index 1  = local tile at grid position (row=0, col=0)
        Index 2  = local tile at (row=0, col=1)
        ...
        Index 16 = local tile at (row=3, col=3)

        Returns a sorted list of tile indices to keep.
        """
        n_local = self.n_local ** 2                           # 16 local tiles
        n_keep = max(self.min_local_tiles,
                     int(n_local * self.keep_ratio))           # how many local tiles to keep

        scores = self.tile_scores(gaze_x, gaze_y, sigma)     # shape (16,)
        _, top_local = torch.topk(scores, k=n_keep)           # best local tile indices (0-15)

        # Shift by 1 to account for the global tile at index 0
        kept_local = (top_local + 1).tolist()
        kept = sorted([0] + kept_local)                        # always include tile 0
        return kept

    def prune(
        self,
        image_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_token_id: int,
        gaze_x: float,
        gaze_y: float,
        sigma: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """
        Prune image_hidden_states and the matching input_ids / attention_mask.

        This is the main function you call at inference time. It:
          1. Figures out which tiles to keep based on gaze
          2. Slices image_hidden_states to only those tiles
          3. Removes the corresponding <image> tokens from input_ids
             so the sequence is actually shorter (→ real speedup)

        Args:
            image_hidden_states: shape (n_tiles, 64, 576) from get_image_features()
            input_ids:           shape (1, seq_len) — the full token sequence
            attention_mask:      shape (1, seq_len)
            image_token_id:      the integer ID for the <image> special token
            gaze_x, gaze_y:      gaze point in [0, 1]

        Returns:
            pruned_hidden_states : (n_kept_tiles, 64, 576)
            new_input_ids        : (1, shorter_seq_len)
            new_attention_mask   : (1, shorter_seq_len)
            kept_tile_indices    : which tile indices survived (for logging)
        """
        kept = self.select_tiles(gaze_x, gaze_y, sigma)
        n_drop = image_hidden_states.shape[0] - len(kept)
        tokens_to_drop = n_drop * self.TOKENS_PER_TILE

        # --- Prune the vision side ---
        pruned_hidden_states = image_hidden_states[kept]   # (n_kept, 64, 576)

        # --- Prune the text sequence side ---
        # Move to CPU for indexing — MPS has limitations with nonzero/advanced indexing
        ids      = input_ids[0].cpu()
        attn_cpu = attention_mask[0].cpu()

        image_mask      = ids == image_token_id
        image_positions = image_mask.nonzero(as_tuple=False).squeeze(1)  # (n_image_tokens,)

        # Drop the LAST tokens_to_drop image positions.
        # (Tile order in the sequence matches tile order in image_hidden_states,
        # so removing tail positions = removing last tiles = the far-from-gaze ones.)
        drop_positions = image_positions[-tokens_to_drop:]
        keep_mask = torch.ones(ids.shape[0], dtype=torch.bool)
        keep_mask[drop_positions] = False

        new_input_ids     = ids[keep_mask].unsqueeze(0)
        new_attention_mask = attn_cpu[keep_mask].unsqueeze(0)

        return pruned_hidden_states, new_input_ids, new_attention_mask, kept


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to make sure nothing is broken
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    print("Running GazePruner sanity check...")

    # SmolVLM-256M splits a 384x384 image into 27x27 patches (patch size = 14)
    # After pixel shuffle it becomes 14x14 = 196 tokens. Let's use that.
    GRID_H, GRID_W = 14, 14
    HIDDEN_DIM = 512
    n_tokens = GRID_H * GRID_W  # 196

    pruner = GazePruner(grid_h=GRID_H, grid_w=GRID_W, keep_ratio=0.5, min_tokens=16)

    # Fake visual tokens (random — just checking shapes)
    fake_tokens = torch.randn(n_tokens, HIDDEN_DIM)

    # User is looking at the center of the image
    gaze_x, gaze_y = 0.5, 0.5

    kept, idx = pruner.prune_by_gaze(fake_tokens, gaze_x, gaze_y)
    print(f"  Original tokens : {n_tokens}")
    print(f"  Kept tokens     : {kept.shape[0]}  (keep_ratio=0.5)")
    print(f"  Dropped tokens  : {n_tokens - kept.shape[0]}")

    # Visualize which patches survive — save a heatmap
    scores = pruner.gaussian_gaze_scores(gaze_x, gaze_y).reshape(GRID_H, GRID_W)
    plt.figure(figsize=(4, 4))
    plt.imshow(scores.numpy(), cmap="hot", vmin=0, vmax=1)
    plt.colorbar(label="token score (1 = keep, 0 = prune)")
    plt.title(f"Gaze at ({gaze_x}, {gaze_y}) — Gaussian falloff")
    plt.xlabel("patch column")
    plt.ylabel("patch row")
    plt.tight_layout()
    plt.savefig("gaze_heatmap.png", dpi=100)
    print("  Heatmap saved → gaze_heatmap.png")

    # Quick test: gaze in top-left corner — top-left patches should score highest
    scores_tl = pruner.gaussian_gaze_scores(0.0, 0.0)
    assert scores_tl[0] > scores_tl[-1], "top-left token should score higher than bottom-right"
    print("  Corner test passed.")

    print("All checks passed!")
