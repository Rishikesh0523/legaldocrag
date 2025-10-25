from sentence_transformers import CrossEncoder


class Reranker:
	"""Reranks documents using a Cross-Encoder model."""
	def __init__(self, model_name: str):
		self.model = CrossEncoder(model_name)
		print(f"Reranker: Loaded Cross-Encoder model '{model_name}'.")

	def rerank(self, query: str, docs: dict, doc_ids: list) -> list:
		"""Reranks a list of documents and returns them sorted by relevance."""
		pairs = [[query, docs[doc_id]] for doc_id in doc_ids]
		scores = self.model.predict(pairs, show_progress_bar=False)

		reranked_results = []
		for i, doc_id in enumerate(doc_ids):
			reranked_results.append({
				"id": doc_id,
				"score": float(scores[i]),
				"text": docs[doc_id],
			})

		# Sort by score in descending order
		reranked_results.sort(key=lambda x: x['score'], reverse=True)
		print(f"Reranker: Reranked {len(reranked_results)} docs. Top score: {reranked_results[0]['score']:.4f}" if reranked_results else "Reranker: No docs to rerank.")
		return reranked_results
