import re


class CitationParser:
	"""Utility to parse and format citations."""
	@staticmethod
	def link_citations(response: str, reranked_docs: list) -> str:
		"""Finds citation markers and adds placeholder links."""
		doc_map = {doc['id']: f"Source: {doc['id']}" for doc in reranked_docs}

		def replace_citation(match):
			doc_id = match.group(1)
			return f"[{doc_id}]({doc_map.get(doc_id, 'Source Not Found')})"

		# Regex to find citations like [doc_id_1]
		linked_response = re.sub(r'\[(doc_id_\d+)\]', replace_citation, response)
		return linked_response
