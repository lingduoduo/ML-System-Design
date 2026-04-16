import heapq
from typing import Dict, List, Tuple

from search_engine import SearchQueryProcessor


class SearchRecallEngine:
    def __init__(self, language: str = "en"):
        self.processor = SearchQueryProcessor(language=language)
        self.main_poi_index: List[Dict[str, object]] = []
        self.backup_poi_index: List[Dict[str, object]] = []
        self.initialize_poi_index()

    def initialize_poi_index(self, poi_index: List[Dict[str, object]] = None) -> None:
        poi_index = poi_index if poi_index is not None else self._default_poi_index()
        prepared_index = [self._prepare_poi(poi) for poi in poi_index]
        self.main_poi_index = [poi for poi in prepared_index if poi.get("partner", False)]
        self.backup_poi_index = [poi for poi in prepared_index if not poi.get("partner", False)]

    def _prepare_poi(self, poi: Dict[str, object]) -> Dict[str, object]:
        prepared = poi.copy()
        prepared["_search_fields"] = {
            "name": str(prepared.get("name", "")).lower(),
            "category": str(prepared.get("category", "")).lower(),
            "city": str(prepared.get("city", "")).lower(),
            "description": str(prepared.get("description", "")).lower(),
        }
        return prepared

    def _default_poi_index(self) -> List[Dict[str, object]]:
        return [
            {
                "id": 1,
                "name": "Tower of London",
                "category": "Historic Landmark",
                "city": "London",
                "description": "Ancient fortress and UNESCO World Heritage Site, home to the Crown Jewels.",
                "partner": True,
            },
            {
                "id": 2,
                "name": "British Museum",
                "category": "Museum",
                "city": "London",
                "description": "World-class museum housing art and artefacts from global cultures, with free admission.",
                "partner": True,
            },
            {
                "id": 3,
                "name": "Tate Modern",
                "category": "Gallery",
                "city": "London",
                "description": "Contemporary art gallery in a converted power station on the South Bank.",
                "partner": False,
            },
            {
                "id": 4,
                "name": "National Portrait Gallery",
                "category": "Gallery",
                "city": "London",
                "description": "Portraits of historically significant people from British history and culture.",
                "partner": False,
            },
        ]

    def _field_match_score(self, poi: Dict[str, object], terms: List[str]) -> float:
        field_weights = {
            "name": 3.0,
            "category": 2.5,
            "city": 2.0,
            "description": 1.0,
        }
        source = poi.get("_search_fields") or {field: str(poi.get(field, "")).lower() for field in field_weights}
        score = 0.0
        for term in terms:
            normalized_term = term.lower()
            for field, weight in field_weights.items():
                if normalized_term in source[field]:
                    score += weight
        if poi.get("partner"):
            score += 0.2
        return score

    def _search_pool(self, terms: List[str], pool: List[Dict[str, object]], top_k: int) -> List[Dict[str, object]]:
        scored = [(self._field_match_score(poi, terms), poi) for poi in pool]
        scored = [(score, poi) for score, poi in scored if score > 0.0]
        ranked = heapq.nlargest(top_k, scored, key=lambda item: item[0])

        results = []
        for score, poi in ranked:
            poi_copy = {key: value for key, value in poi.items() if key != "_search_fields"}
            poi_copy["recall_score"] = round(score, 4)
            results.append(poi_copy)
        return results

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

    def recall(self, query: str, max_chars: int = 64, top_k: int = 5) -> Dict[str, object]:
        query_info = self.processor.process_query(query, max_chars=max_chars)
        query_info["search_terms"] = self._expand_search_terms(query_info["lemmas"])

        recall_steps: List[str] = []
        main_hits = self._search_pool(query_info["search_terms"], self.main_poi_index, top_k=top_k)
        recall_steps.append("main_pool_recall")

        if main_hits:
            query_info["recall"] = self._build_recall_result(main_hits, [], main_hits, recall_steps)
            return query_info

        backup_hits = self._search_pool(query_info["search_terms"], self.backup_poi_index, top_k=top_k)
        recall_steps.append("backup_pool_recall")

        if backup_hits:
            query_info["recall"] = self._build_recall_result([], backup_hits, backup_hits, recall_steps)
            return query_info

        if query_info["category_intent"] == "category_search":
            reduced_terms = self._drop_modifiers(query_info["lemmas"], query_info["chunk_tags"])
            reduced_terms = self._expand_search_terms(reduced_terms)
            recall_steps.append("modifier_drop_recall")
            third_hits = self._search_pool(
                reduced_terms,
                self.main_poi_index + self.backup_poi_index,
                top_k=top_k,
            )
            query_info["search_terms"] = reduced_terms
            query_info["recall"] = self._build_recall_result([], [], third_hits, recall_steps)
            return query_info

        query_info["recall"] = self._build_recall_result([], [], [], recall_steps)
        return query_info


if __name__ == "__main__":
    engine = SearchRecallEngine()
    engine.processor.set_idf({"london": 0.48, "famous": 0.39, "museum": 0.55, "historic": 0.55})
    engine.processor.update_click_log(["london", "famous", "museum", "historic"])

    query = "London famous museum"
    result = engine.recall(query)

    print("Query:", query)
    print("Chunk tags:", result["chunk_tags"])
    print("Category intent:", result["category_intent"])
    print("Search terms:", result["search_terms"])
    print("Recall steps:", result["recall"]["recall_steps"])
    print("Final recall:")
    for poi in result["recall"]["final_recall"]:
        print("-", poi["name"], f"({poi['city']} - {poi['category']}, score={poi['recall_score']})")
