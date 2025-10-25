class CorrectiveLayer:
	"""Performs a final consistency and hallucination check."""
	def __init__(self, generator):
		# Uses the same generator LLM for the check, but could use a separate one.
		self.checker_llm = generator
		print("CorrectiveLayer: Initialized.")

	def check(self, response: str, context: str) -> dict:
		"""Checks the response for hallucinations against the context."""
		print("CorrectiveLayer: Performing hallucination check...")
		# Simple logic for demonstration
		raw_check_output = self.checker_llm.generate(
			"Is the following statement fully supported by the provided context?",
			f"Statement: '{response}'\n\nContext: '{context}'",
		)
		is_consistent = "yes" in raw_check_output.lower()
		result = {
			"is_consistent": is_consistent,
			"reasoning": raw_check_output,
		}
		print(f"CorrectiveLayer: Check complete. Consistent: {is_consistent}")
		return result
