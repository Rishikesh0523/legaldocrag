import spacy


class PreprocessingEngine:
	"""Handles NER and Jurisdiction Detection."""
	def __init__(self):
		# Using a general-purpose NER model. For better performance,
		# this should be replaced with a model fine-tuned on legal text.
		self.nlp = spacy.load("en_core_web_sm")
		print("PreprocessingEngine: spaCy NER model loaded.")

	def extract_entities(self, query: str) -> list:
		"""Extracts named entities from the query."""
		doc = self.nlp(query)
		entities = [ent.text for ent in doc.ents]
		print(f"PreprocessingEngine: Extracted entities -> {entities}")
		return entities
