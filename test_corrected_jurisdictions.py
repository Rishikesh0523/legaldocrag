import os
import torch
from legaldocrag import PipelineConfig, LegalRAGPipeline

def main():
    print("=" * 80)
    print("Legal RAG Pipeline v2.0 - Fixed Jurisdiction Detection")
    print("=" * 80)
    
    # Create config with Claude and CORRECTED jurisdiction detection
    config = PipelineConfig()
    config.GENERATOR_TYPE = 'claude'  # Use Claude API
    config.ENABLE_JURISDICTION_DETECTION = True  # Now it should work correctly
    
    print(f"✓ Using Claude API")
    print(f"✓ Jurisdiction detection: ENABLED (with smart ID-based classification)")
    
    # Initialize pipeline
    pipeline = LegalRAGPipeline(config)
    
    # Load documents
    print("\n--- Loading Documents ---")
    pipeline.process_documents(load_from_datasets=True)
    
    # Show correct statistics
    stats = pipeline.document_loader.get_statistics()
    print("\n📊 CORRECTED Document Statistics:")
    print(f"   • Total Documents: {stats['total_documents']}")
    
    if stats['by_jurisdiction']:
        print(f"\n   By Jurisdiction (SHOULD BE CORRECT NOW):")
        for jurisdiction, count in sorted(stats['by_jurisdiction'].items()):
            print(f"     - {jurisdiction.capitalize()}: {count}")
    
    # Test queries for each jurisdiction
    test_queries = [
        {
            "query": "What does Mexican Civil Law say about contracts?",
            "expected_jurisdiction": "mexico"
        },
        {
            "query": "What fundamental rights are guaranteed in the Constitution of Nepal?", 
            "expected_jurisdiction": "nepal"
        }
    ]
    
    print("\n" + "="*60)
    print("TESTING JURISDICTION-SPECIFIC QUERIES")
    print("="*60)
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {test['expected_jurisdiction'].upper()} Query ---")
        print(f"Query: {test['query']}")
        
        result = pipeline.answer_query(test['query'])
        
        print(f"\nSuccess: {result['success']}")
        if result.get('jurisdictions_detected'):
            detected = result['jurisdictions_detected']
            expected = test['expected_jurisdiction']
            status = "✅ CORRECT" if expected in detected else "❌ INCORRECT"
            print(f"Jurisdictions Detected: {detected} {status}")
        
        print(f"Answer: {result['answer'][:200]}...")
        
        if result.get('sources'):
            print(f"\nTop Sources:")
            for source in result['sources'][:2]:
                jurisdiction = pipeline.document_loader.get_metadata(source['id']).get('jurisdiction', 'unknown')
                print(f"  - {source['id']} (jurisdiction: {jurisdiction})")

if __name__ == "__main__":
    main()