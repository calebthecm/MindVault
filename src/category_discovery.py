"""
Category discovery: finds new topic clusters in conversations that don't fit
existing categories well, and proposes (or auto-creates) new ones.

Algorithm:
  1. Score every conversation against existing CATEGORY_RULES (keyword hit count)
  2. Conversations below WEAK_MATCH_THRESHOLD go into the discovery pool
  3. Extract meaningful keywords from the pool (title weighted 3x, summary 1x)
  4. Build a keyword → [conversation_uuids] index
  5. Cluster conversations that share 2+ keywords
  6. If a cluster reaches VOTE_THRESHOLD, it earns a new category
  7. New category gets: a folder name, a tag, and a color from DISCOVERY_PALETTE
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# ─── Tuning knobs ─────────────────────────────────────────────────────────────

# A conversation is "weakly matched" if it hits fewer than this many keywords
# across all existing rules combined. Lower = more goes into discovery pool.
WEAK_MATCH_THRESHOLD = 2

# Minimum conversations a cluster needs to earn a new category.
# Also must be >= MIN_CLUSTER_PCT % of total conversations.
MIN_CLUSTER_SIZE = 3
MIN_CLUSTER_PCT = 0.06  # 6%

# Minimum number of conversations two nodes share a keyword for it to count
MIN_KEYWORD_SUPPORT = 2

# Colors for dynamically discovered categories (decimal RGB)
DISCOVERY_PALETTE = [
    ("Cyan",        4169216),   # #3fb500 — neon green ... actually:
    ("Cyan",        3182592),   # #309000
    ("Slate",       7108344),   # #6c7bc8
    ("Rose",        14696960),  # #e04040
    ("Lime",        5066061),   # #4d7b4d ... let me use real values
]

# Better palette — distinct from the 9 existing colors
DISCOVERY_PALETTE = [
    ("Cyan",     3386060),   # #33b4cc
    ("Lime",     5991234),   # #5b8d02  — no let me compute these properly
]

# Recomputed properly:
# #06b6d4 cyan:      r=6,   g=182, b=212 → 6*65536+182*256+212 = 393216+46592+212 = 440020
# #84cc16 lime:      r=132, g=204, b=22  → 132*65536+204*256+22 = 8650752+52224+22 = 8702998
# #f43f5e rose:      r=244, g=63,  b=94  → 244*65536+63*256+94  = 15990784+16128+94 = 16007006
# #0ea5e9 sky:       r=14,  g=165, b=233 → 14*65536+165*256+233 = 917504+42240+233 = 959977
# #a855f7 violet:    r=168, g=85,  b=247 → 168*65536+85*256+247 = 11010048+21760+247 = 11032055
# #10b981 emerald:   r=16,  g=185, b=129 → 16*65536+185*256+129 = 1048576+47360+129 = 1096065
# #f59e0b already used (amber)
# #64748b slate:     r=100, g=116, b=139 → 100*65536+116*256+139 = 6553600+29696+139 = 6583435

DISCOVERY_PALETTE = [
    ("Cyan",    440020),
    ("Lime",    8702998),
    ("Rose",    16007006),
    ("Sky",     959977),
    ("Violet",  11032055),
    ("Emerald", 1096065),
    ("Slate",   6583435),
]


STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "it", "its", "how", "what", "when", "where", "why", "who",
    "which", "my", "your", "his", "her", "our", "their", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "same", "so", "than", "too", "very", "just", "about",
    "into", "through", "during", "before", "after", "above", "use", "using",
    "used", "create", "creating", "build", "building", "add", "new", "review",
    "update", "based", "also", "well", "like", "make", "made", "get", "got",
    "set", "up", "out", "one", "two", "three", "including", "overview",
    "conversation", "asked", "provided", "claude", "system",
    "simple", "single", "brief", "quickly", "overall", "then", "been",
    "first", "second", "third", "across", "within", "without", "between",
}

# Meaningful bigrams to extract as single tokens
BIGRAM_PATTERNS = [
    "local business", "business automation", "skill creator", "agent maker",
    "compound interest", "privacy policy", "email filter", "lead generation",
    "yield sign", "invoice app", "scheduling app", "cleaning service",
    "graph view", "obsidian vault", "notion database",
]


# ─── Scoring existing rules ───────────────────────────────────────────────────

def score_conversation(title: str, summary: str, category_rules: list) -> int:
    """Count how many existing-rule keywords match this conversation."""
    combined = (title + " " + summary).lower()
    hits = 0
    for keywords, _cat, _tags in category_rules:
        hits += sum(1 for kw in keywords if kw in combined)
    return hits


# ─── Keyword extraction ───────────────────────────────────────────────────────

def extract_keywords(text: str) -> list[str]:
    """Extract meaningful single words and bigrams from text."""
    text = text.lower()
    keywords = []

    # Extract bigrams first
    for bigram in BIGRAM_PATTERNS:
        if bigram in text:
            keywords.append(bigram.replace(" ", "-"))
            text = text.replace(bigram, "")  # don't double-count words in bigrams

    # Extract words
    words = re.findall(r"\b[a-z][a-z0-9]{2,}\b", text)
    for word in words:
        if word not in STOP_WORDS and len(word) > 3:
            keywords.append(word)

    return keywords


def keyword_frequency(conversations: list[dict]) -> dict[str, list[str]]:
    """
    Returns {keyword: [conv_uuid, ...]} for keywords that appear in
    at least MIN_KEYWORD_SUPPORT conversations.
    """
    keyword_to_convos: dict[str, list[str]] = defaultdict(list)

    for convo in conversations:
        uuid = convo["uuid"]
        title = convo.get("name", "")
        summary = convo.get("summary", "")

        # Title words weighted 3x by repeating them
        title_kws = extract_keywords(title) * 3
        summary_kws = extract_keywords(summary)
        seen = set()
        for kw in title_kws + summary_kws:
            if kw not in seen:
                keyword_to_convos[kw].append(uuid)
                seen.add(kw)

    # Filter to keywords with minimum support
    return {
        kw: uuids
        for kw, uuids in keyword_to_convos.items()
        if len(uuids) >= MIN_KEYWORD_SUPPORT
    }


# ─── Clustering ───────────────────────────────────────────────────────────────

def cluster_by_keywords(
    pool: list[dict],
    keyword_index: dict[str, list[str]],
) -> list[tuple[str, list[str], list[dict]]]:
    """
    Group conversations into clusters based on shared keywords.
    Returns list of (representative_keyword, [keywords], [conversations]).
    """
    uuid_to_convo = {c["uuid"]: c for c in pool}

    # Build conversation → keyword set
    convo_keywords: dict[str, set[str]] = defaultdict(set)
    for kw, uuids in keyword_index.items():
        for uuid in uuids:
            if uuid in uuid_to_convo:
                convo_keywords[uuid].add(kw)

    # Group conversations that share the most keywords
    # Simple greedy approach: sort keywords by support, assign convos to first matching cluster
    sorted_keywords = sorted(keyword_index.items(), key=lambda x: len(x[1]), reverse=True)

    assigned: set[str] = set()
    clusters: list[tuple[str, list[str], list[dict]]] = []

    for seed_kw, seed_uuids in sorted_keywords:
        # Only use pool UUIDs
        pool_uuids = [u for u in seed_uuids if u in uuid_to_convo and u not in assigned]
        if len(pool_uuids) < MIN_CLUSTER_SIZE:
            continue

        # Expand cluster: find all pool convos that share >= 1 keyword with seed members
        cluster_uuids = set(pool_uuids)
        shared_keywords = {seed_kw}

        for kw, kw_uuids in keyword_index.items():
            if kw == seed_kw:
                continue
            overlap = set(kw_uuids) & cluster_uuids
            if len(overlap) >= max(2, len(cluster_uuids) // 2):
                shared_keywords.add(kw)

        cluster_convos = [uuid_to_convo[u] for u in cluster_uuids]
        clusters.append((seed_kw, sorted(shared_keywords), cluster_convos))
        assigned.update(cluster_uuids)

    return clusters


# ─── Category name generation ─────────────────────────────────────────────────

def make_category_name(keywords: list[str]) -> str:
    """Turn a list of keywords into a readable category name."""
    # Prefer multi-word keywords (bigrams) as they're more descriptive
    bigrams = [kw for kw in keywords if "-" in kw]
    singles = [kw for kw in keywords if "-" not in kw]

    if bigrams:
        name = bigrams[0].replace("-", " ").title()
    elif singles:
        # Use the top 2 most-meaningful single words
        name = " & ".join(w.title() for w in singles[:2])
    else:
        name = "General"

    return name


def make_tag(category_name: str) -> str:
    return re.sub(r"[^\w]+", "-", category_name.lower()).strip("-")


# ─── Main discovery function ──────────────────────────────────────────────────

def discover_categories(
    conversations: list[dict],
    category_rules: list,
    existing_category_names: set[str],
    palette_offset: int = 0,
) -> list[dict]:
    """
    Analyze conversations and discover new categories.

    Returns list of new category dicts:
    {
        "name": str,          # folder name e.g. "Local Business"
        "tag": str,           # tag slug e.g. "local-business"
        "color_rgb": int,     # decimal RGB for graph.json
        "color_name": str,    # human label
        "conversations": [...] # list of conversation dicts
        "keywords": [...]
    }
    """
    total = len(conversations)
    threshold = max(MIN_CLUSTER_SIZE, int(total * MIN_CLUSTER_PCT))

    # Step 1: Score each conversation, collect weak matches into pool
    pool = []
    for convo in conversations:
        title = convo.get("name", "")
        summary = convo.get("summary", "")
        score = score_conversation(title, summary, category_rules)
        if score < WEAK_MATCH_THRESHOLD:
            pool.append(convo)

    print(f"\n[Discovery] {len(pool)} of {total} conversations entered discovery pool (score < {WEAK_MATCH_THRESHOLD} keyword hits)")

    if not pool:
        print("[Discovery] Pool is empty — all conversations matched existing categories well.")
        return []

    # Step 2: Keyword frequency in the pool
    keyword_index = keyword_frequency(pool)
    if not keyword_index:
        print("[Discovery] No significant keyword clusters found in pool.")
        return []

    print(f"[Discovery] Found {len(keyword_index)} candidate keywords in pool")

    # Step 3: Cluster
    clusters = cluster_by_keywords(pool, keyword_index)

    # Step 4: Vote — only clusters that hit the threshold earn a new category
    new_categories = []
    palette_index = palette_offset % len(DISCOVERY_PALETTE)

    for seed_kw, keywords, cluster_convos in clusters:
        votes = len(cluster_convos)
        name = make_category_name(keywords)
        tag = make_tag(name)

        # Don't create if name too close to existing category
        if any(name.lower() in ex.lower() or ex.lower() in name.lower()
               for ex in existing_category_names):
            print(f"[Discovery] '{name}' ({votes} convos) — skipped, similar to existing category")
            continue

        if votes < threshold:
            print(f"[Discovery] '{name}' ({votes} convos, keywords: {keywords[:4]}) — below threshold ({threshold}), not promoted")
            continue

        color_name, color_rgb = DISCOVERY_PALETTE[palette_index % len(DISCOVERY_PALETTE)]
        palette_index += 1

        print(f"[Discovery] NEW CATEGORY: '{name}' — {votes} conversations, tag: #{tag}, color: {color_name}")
        for c in cluster_convos:
            print(f"             - {c['name']}")

        new_categories.append({
            "name": name,
            "tag": tag,
            "color_rgb": color_rgb,
            "color_name": color_name,
            "conversations": cluster_convos,
            "keywords": keywords,
        })

    if not new_categories:
        print("[Discovery] No clusters passed the vote threshold. No new categories created.")

    return new_categories


def add_colors_to_graph(new_categories: list[dict], graph_json_path: Path) -> None:
    """Append new category color groups to Obsidian graph.json."""
    if not new_categories or not graph_json_path.exists():
        return

    with open(graph_json_path) as f:
        graph = json.load(f)

    existing_queries = {g["query"] for g in graph.get("colorGroups", [])}

    added = 0
    for cat in new_categories:
        query = f"tag:#{cat['tag']}"
        if query in existing_queries:
            continue
        graph["colorGroups"].append({
            "query": query,
            "color": {"a": 1, "rgb": cat["color_rgb"]},
        })
        existing_queries.add(query)
        added += 1

    if added:
        with open(graph_json_path, "w") as f:
            json.dump(graph, f, indent=2)
        print(f"[Discovery] Added {added} new color group(s) to graph.json")
