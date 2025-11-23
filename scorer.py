# scorer placeholder
# scorer.py
import math
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize

MODEL_NAME = "all-MiniLM-L6-v2"

class RubricScorer:
    def __init__(self, rubric_excel_path, model_name=MODEL_NAME):
        self.rubric = pd.read_excel(rubric_excel_path)
        self.model = SentenceTransformer(model_name)
        # Expect rubric columns: 'criterion_id','criterion_name','description','keywords','weight','min_words','max_words'
        self.normalize_weights()

    def normalize_weights(self):
        if 'weight' in self.rubric.columns:
            total = self.rubric['weight'].sum()
            if total == 0:
                self.rubric['norm_weight'] = 1.0 / len(self.rubric)
            else:
                self.rubric['norm_weight'] = self.rubric['weight'] / total
        else:
            self.rubric['norm_weight'] = 1.0 / len(self.rubric)

    def _keywords_list(self, kw_string):
        if pd.isna(kw_string):
            return []
        # assume keywords comma-separated
        kws = [k.strip().lower() for k in re.split(r'[,;|]', str(kw_string)) if k.strip()]
        return kws

    def compute_scores(self, transcript_text,
                       w_rule=0.4, w_sem=0.5, w_len=0.1):
        transcript = transcript_text.strip()
        words = word_tokenize(transcript)
        n_words = len(words)
        # Precompute embedding for whole transcript and individual sentences
        sent_list = sent_tokenize(transcript)
        if len(sent_list)==0:
            sent_list = [transcript]
        sent_embeddings = self.model.encode(sent_list, convert_to_numpy=True)
        transcript_embedding = np.mean(sent_embeddings, axis=0, keepdims=True)

        results = []
        raw_score = 0.0

        for idx, row in self.rubric.iterrows():
            crit_id = row.get('criterion_id', idx)
            crit_name = row.get('criterion_name', row.get('description', f'criterion_{idx}'))
            description = row.get('description', '')
            keywords = self._keywords_list(row.get('keywords', ''))
            norm_w = float(row.get('norm_weight', 1.0))
            min_w = row.get('min_words', np.nan)
            max_w = row.get('max_words', np.nan)

            # Rule-based: keyword matches
            matched = []
            if keywords:
                for kw in keywords:
                    # simple substring match (case-insensitive)
                    if re.search(r'\b' + re.escape(kw) + r'\b', transcript, flags=re.I):
                        matched.append(kw)
                S_rule = len(matched) / max(1, len(keywords))
            else:
                S_rule = 0.5  # neutral if no keywords specified

            # Semantic: compare description embedding to transcript sentences; take max
            desc_emb = self.model.encode([description], convert_to_numpy=True)
            sims = cosine_similarity(desc_emb, transcript_embedding)[0]
            # sims is array length 1
            sim_val = float(sims[0])
            S_sem = (sim_val + 1.0) / 2.0  # map -1..1 -> 0..1

            # Length
            if not (math.isnan(min_w) and math.isnan(max_w)):
                # if min or max not provided, adapt
                min_w_eff = min_w if not math.isnan(min_w) else 0
                max_w_eff = max_w if not math.isnan(max_w) else 1e9
                if min_w_eff <= n_words <= max_w_eff:
                    S_len = 1.0
                else:
                    nearest = min_w_eff if n_words < min_w_eff else max_w_eff
                    S_len = max(0.0, 1.0 - (abs(n_words - nearest) / max(20.0, nearest)))
            else:
                S_len = 1.0  # neutral

            S_comb = w_rule * S_rule + w_sem * S_sem + w_len * S_len
            # per-criterion score out of 100 scaled by its normalized weight
            per_criterion_score_0_100 = S_comb * 100
            raw_score += S_comb * norm_w

            feedback = []
            if keywords:
                feedback.append(f"Keywords matched: {matched}" if matched else "No rubric keywords matched.")
            feedback.append(f"Semantic similarity (mapped 0-1): {S_sem:.3f}")
            if not (math.isnan(min_w) and math.isnan(max_w)):
                feedback.append(f"Length: {n_words} words (expected {min_w}-{max_w})." if not (min_w <= n_words <= max_w) else "Length within expected range.")

            results.append({
                "criterion_id": str(crit_id),
                "criterion_name": str(crit_name),
                "weight": float(norm_w),
                "S_rule": round(S_rule, 3),
                "S_sem": round(S_sem, 3),
                "S_len": round(S_len, 3),
                "S_combined": round(S_comb, 3),
                "score_out_of_100": round(per_criterion_score_0_100, 1),
                "matched_keywords": matched,
                "feedback": " ".join(feedback)
            })

        overall = round(raw_score * 100, 1)
        return {
            "overall_score": overall,
            "words": n_words,
            "per_criterion": results
        }
