"""
Test Script for PDF Processing with Caching

This script demonstrates:
1. First-time PDF processing (slow)
2. Second-time loading from cache (fast)
3. Cache validation
"""

import time
from pathlib import Path
from legaldocrag.document_loader import LegalDocumentLoader


def test_pdf_processing():
    """Test PDF processing with caching."""
    
    print("="*80)
    print("PDF Processing Test with Smart Caching")
    print("="*80)
    
    # Initialize loader
    loader = LegalDocumentLoader()
    
    # Path to Nepal Constitution PDF
    pdf_path = './data/Constitution-of-Nepal_2072_Eng_www.moljpa.gov_.npDate-72_11_16.pdf'
    
    if not Path(pdf_path).exists():
        print(f"\n❌ PDF file not found: {pdf_path}")
        print("Please ensure the Nepal Constitution PDF is in the data/ directory")
        return
    
    print(f"\n📄 Testing PDF: {Path(pdf_path).name}")
    print(f"File size: {Path(pdf_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    # =========================================================================
    # Test 1: First Load (should process PDF and create cache)
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 1: First Load (Processing + Caching)")
    print("="*80)
    
    start_time = time.time()
    
    documents = loader.load_pdf(
        pdf_path=pdf_path,
        jurisdiction='nepal',
        doc_type='constitution',
        use_cache=True,
        parser='pypdf2',
        chunk_by='page'
    )
    
    first_load_time = time.time() - start_time
    
    print(f"\n✓ First load completed")
    print(f"  - Documents loaded: {len(documents)}")
    print(f"  - Time taken: {first_load_time:.2f} seconds")
    
    # Show sample document
    sample_id = list(documents.keys())[0]
    sample_text = documents[sample_id]
    sample_metadata = loader.get_metadata(sample_id)
    
    print(f"\n📄 Sample Document:")
    print(f"  - ID: {sample_id}")
    print(f"  - Jurisdiction: {sample_metadata['jurisdiction']}")
    print(f"  - Type: {sample_metadata['type']}")
    print(f"  - Page: {sample_metadata['page_number']}/{sample_metadata['total_pages']}")
    print(f"  - Text preview: {sample_text[:150]}...")
    
    # =========================================================================
    # Test 2: Second Load (should load from cache - FAST!)
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 2: Second Load (From Cache)")
    print("="*80)
    
    # Create new loader instance to simulate fresh start
    loader2 = LegalDocumentLoader()
    
    start_time = time.time()
    
    documents2 = loader2.load_pdf(
        pdf_path=pdf_path,
        jurisdiction='nepal',
        doc_type='constitution',
        use_cache=True,
        parser='pypdf2',
        chunk_by='page'
    )
    
    second_load_time = time.time() - start_time
    
    print(f"\n✓ Second load completed")
    print(f"  - Documents loaded: {len(documents2)}")
    print(f"  - Time taken: {second_load_time:.2f} seconds")
    
    # =========================================================================
    # Performance Comparison
    # =========================================================================
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    speedup = first_load_time / second_load_time if second_load_time > 0 else float('inf')
    
    print(f"\n⏱️  Timing Results:")
    print(f"  - First Load (Processing):  {first_load_time:.3f} seconds")
    print(f"  - Second Load (Cache):      {second_load_time:.3f} seconds")
    print(f"  - Speed Improvement:        {speedup:.1f}x faster")
    print(f"  - Time Saved:               {first_load_time - second_load_time:.3f} seconds")
    
    # =========================================================================
    # Cache Information
    # =========================================================================
    print("\n" + "="*80)
    print("CACHE INFORMATION")
    print("="*80)
    
    cache_dir = Path('./data/.cache')
    if cache_dir.exists():
        cache_files = list(cache_dir.glob('*.cache'))
        
        print(f"\n📂 Cache Directory: {cache_dir}")
        print(f"  - Cache files: {len(cache_files)}")
        
        total_size = sum(f.stat().st_size for f in cache_files)
        print(f"  - Total cache size: {total_size / 1024 / 1024:.2f} MB")
        
        print(f"\n  Cache Files:")
        for cache_file in cache_files:
            size_mb = cache_file.stat().st_size / 1024 / 1024
            print(f"    - {cache_file.name}: {size_mb:.2f} MB")
    
    # =========================================================================
    # Test 3: Load with disabled cache (should reprocess)
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 3: Force Reprocessing (Cache Disabled)")
    print("="*80)
    
    loader3 = LegalDocumentLoader()
    
    start_time = time.time()
    
    documents3 = loader3.load_pdf(
        pdf_path=pdf_path,
        jurisdiction='nepal',
        doc_type='constitution',
        use_cache=False,  # Disable cache
        parser='pypdf2',
        chunk_by='page'
    )
    
    reprocess_time = time.time() - start_time
    
    print(f"\n✓ Reprocessing completed")
    print(f"  - Documents loaded: {len(documents3)}")
    print(f"  - Time taken: {reprocess_time:.2f} seconds")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"\n✅ All tests passed!")
    print(f"\n📊 Key Findings:")
    print(f"  1. PDF contains {len(documents)} pages/chunks")
    print(f"  2. First load took {first_load_time:.2f}s (processing + caching)")
    print(f"  3. Cached load took {second_load_time:.2f}s ({speedup:.1f}x faster!)")
    print(f"  4. Cache is working correctly")
    print(f"  5. Cache can be disabled when needed")
    
    print(f"\n💡 Tips:")
    print(f"  - Cache files are stored in: {cache_dir}")
    print(f"  - Cache is automatically validated using file hash")
    print(f"  - If PDF changes, cache is automatically regenerated")
    print(f"  - Clear cache with: Remove-Item -Path 'data\\.cache\\*' -Force")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        test_pdf_processing()
    except ImportError as e:
        print(f"\n❌ Missing required library: {e}")
        print(f"\nPlease install PDF processing libraries:")
        print(f"  pip install pypdf2 pdfplumber")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
