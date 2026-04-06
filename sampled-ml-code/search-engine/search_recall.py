from typing import List, Dict, Tuple

from query_understanding import SearchQueryProcessor


class SearchRecallEngine:
    def __init__(self, language: str = "en"):
        self.processor = SearchQueryProcessor(language=language)
        self.main_poi_index: List[Dict[str, object]] = []
        self.backup_poi_index: List[Dict[str, object]] = []
        self.initialize_poi_index()

    def initialize_poi_index(self, poi_index: List[Dict[str, object]] = None) -> None:
        poi_index = poi_index if poi_index is not None else self._default_poi_index()
        self.main_poi_index = [poi for poi in poi_index if poi.get("partner", False)]
        self.backup_poi_index = [poi for poi in poi_index if not poi.get("partner", False)]

    def _default_poi_index(self) -> List[Dict[str, object]]:
        return [
            ...
        ]

    def _field_match_score(self, poi: Dict[str, object], terms: List[str]) -> float:
        field_weights = {
            "name": 3.0,
            "category": 2.5,
            "city": 2.0,
            "description": 1.0,
        }
        source = {field: str(poi.get(field, "")).lower() for field in field_weights}
        return sum(
            weight
            for term in terms
            for field, weight in field_weights.items()
            if term.lower() in source[field]
        )

    def _search_pool(self, terms: List[str], pool: List[Dict[str, object]]) -> List[Dict[str, object]]:
        scored = [(self._field_match_score(poi, terms), poi) for poi in pool]
        scored = [(score, poi) for score, poi in scored if score > 0.0]
        return [poi for score, poi in sorted(scored, key=lambda item: item[0], reverse=True)]

    def _drop_modifiers(self, lemmas: List[str], chunk_tags: List[Tuple[str, str]]) -> List[str]:
        modifiers = {token.lower() for token, tag in chunk_tags if tag == "MODIFIER"}
        return [term for term in lemmas if term.lower() not in modifiers]

    def _expand_search_terms(self, lemmas: List[str]) -> List[str]:
        return list(dict.fromkeys(lemmas + self.processor.expand_synonyms(lemmas)))

    def _build_recall_result(
        self,
        main_recall: List[Dict[str, object]],
        backup_recall: List[Dict[str, object]],
        final_recall: List[Dict[str, object]],
        recall_steps: List[str],
    ) -> Dict[str, object]:
        return {
            "main_recall": main_recall,
            "backup_recall": backup_recall,
            "final_recall": final_recall,
            "recall_steps": recall_steps,
        }

    def recall(self, query: str, max_chars: int = 64) -> Dict[str, object]:
        query_info = self.processor.process_query(query, max_chars=max_chars)
        query_info["chunk_tags"] = self.processor.analyze_chunks(query_info["tokens"], query_info["pos_tags"])
        query_info["category_intent"] = self.processor.detect_category_intent(query_info["chunk_tags"])
        query_info["search_terms"] = self._expand_search_terms(query_info["lemmas"])

        recall_steps: List[str] = []
        main_hits = self._search_pool(query_info["search_terms"], self.main_poi_index)
        recall_steps.append("main_pool_recall")

        if main_hits:
            query_info["recall"] = self._build_recall_result(main_hits, [], main_hits, recall_steps)
            return query_info

        backup_hits = self._search_pool(query_info["search_terms"], self.backup_poi_index)
        recall_steps.append("backup_pool_recall")

        if backup_hits:
            query_info["recall"] = self._build_recall_result([], backup_hits, backup_hits, recall_steps)
            return query_info

        if query_info["category_intent"] == "category_search":
            reduced_terms = self._drop_modifiers(query_info["lemmas"], query_info["chunk_tags"])
            reduced_terms = self._expand_search_terms(reduced_terms)
            recall_steps.append("modifier_drop_recall")
            third_hits = self._search_pool(reduced_terms, self.main_poi_index + self.backup_poi_index)
            query_info["recall"] = self._build_recall_result([], [], third_hits, recall_steps)
            return query_info

        query_info["recall"] = self._build_recall_result([], [], [], recall_steps)
        return query_info


if __name__ == "__main__":
    engine = SearchRecallEngine()
    engine.processor.set_idf({"new jersey": 0.48, "famous": 0.39, "hot": 0.55, "spring": 0.55})
    engine.processor.update_click_log(["new jersey", "famous", "hot", "spring"])

    query = "New Jersey famous hot spring"
    result = engine.recall(query)

    print("Query:", query)
    print("Chunk tags:", result["chunk_tags"])
    print("Category intent:", result["category_intent"])
    print("Search terms:", result["search_terms"])
    print("Recall steps:", result["recall"]["recall_steps"])
    print("Final recall:")
    for poi in result["recall"]["final_recall"]:
        print("-", poi["name"], f"({poi['city']} - {poi['category']})")
