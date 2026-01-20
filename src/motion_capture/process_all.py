"""
Batch Motion Extraction Processor

Processes all videos in motion_dataset.json and extracts motion data using MediaPipe Holistic.
Generates motion library with one JSON file per gloss.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import traceback
from extract_motion import MotionExtractor


class BatchProcessor:
    """Batch process videos from motion dataset."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize Batch Processor.
        
        Args:
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        self.motion_capture_path = Path(__file__).parent
        self.dataset_path = self.motion_capture_path / "motion_dataset.json"
        self.output_dir = self.motion_capture_path / "motion_library"
        self.base_video_path = self.motion_capture_path.parent / "model" / "finetune" / "data_engineering"
        
        self.dataset: List[Dict] = []
        self.extractor = MotionExtractor(target_fps=30, verbose=verbose)
        
        self.successful_count = 0
        self.failed_count = 0
        self.failed_glosses = []
        
    def load_dataset(self):
        """Load motion dataset from JSON."""
        if self.verbose:
            print("\n📂 Loading motion dataset...")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Motion dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        if self.verbose:
            print(f"   Found {len(self.dataset)} glosses to process")
    
    def process_video(self, entry: Dict) -> bool:
        """
        Process single video and save motion data.
        
        Args:
            entry: Dataset entry with gloss and video_path
            
        Returns:
            True if successful, False otherwise
        """
        gloss = entry['gloss']
        video_path = self.base_video_path / entry['video_path']
        
        try:
            # Check if video exists
            if not video_path.exists():
                if self.verbose:
                    print(f"\n❌ Video not found: {video_path}")
                return False
            
            # Extract motion data
            motion_data = self.extractor.extract_motion(str(video_path), gloss)
            
            # Save to motion library
            output_path = self.output_dir / f"{gloss.upper()}.json"
            self.extractor.save_motion_data(motion_data, output_path)
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"\n❌ Error processing {gloss}: {str(e)}")
                if '--debug' in sys.argv:
                    traceback.print_exc()
            return False
    
    def process_all(self, limit: int = None):
        """
        Process all videos in dataset.
        
        Args:
            limit: Optional limit on number of videos to process (for testing)
        """
        dataset_to_process = self.dataset[:limit] if limit else self.dataset
        total = len(dataset_to_process)
        
        if self.verbose:
            print(f"\n🚀 Starting batch processing ({total} videos)...")
            print("="*60)
        
        for idx, entry in enumerate(dataset_to_process, 1):
            if self.verbose:
                print(f"\n[{idx}/{total}] Processing: {entry['gloss'].upper()}")
            
            success = self.process_video(entry)
            
            if success:
                self.successful_count += 1
            else:
                self.failed_count += 1
                self.failed_glosses.append(entry['gloss'])
        
        # Cleanup
        self.extractor.close()
    
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "="*60)
        print("📊 BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total videos:          {len(self.dataset)}")
        print(f"Successfully processed: {self.successful_count}")
        print(f"Failed:                {self.failed_count}")
        print(f"Success rate:          {self.successful_count / len(self.dataset) * 100:.1f}%")
        print("="*60)
        
        if self.failed_glosses:
            print(f"\n⚠️  Failed glosses ({len(self.failed_glosses)}):")
            for gloss in self.failed_glosses[:20]:
                print(f"   - {gloss}")
            if len(self.failed_glosses) > 20:
                print(f"   ... and {len(self.failed_glosses) - 20} more")
        
        print(f"\n✅ Motion library saved to: {self.output_dir}")
    
    def run(self, limit: int = None):
        """
        Execute complete batch processing pipeline.
        
        Args:
            limit: Optional limit for testing (e.g., 5 for first 5 videos)
        """
        print("\n" + "="*60)
        print("🎬 BATCH MOTION EXTRACTION PIPELINE")
        print("="*60)
        
        try:
            self.load_dataset()
            self.process_all(limit=limit)
            self.print_summary()
            
            print("\n✅ Batch processing completed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Fatal error during batch processing: {e}")
            if '--debug' in sys.argv:
                traceback.print_exc()
            return False


def main():
    """Main entry point."""
    # Check for test mode (process only first N videos)
    limit = None
    if '--test' in sys.argv:
        try:
            test_idx = sys.argv.index('--test')
            limit = int(sys.argv[test_idx + 1]) if len(sys.argv) > test_idx + 1 else 5
            print(f"\n🧪 TEST MODE: Processing first {limit} videos only")
        except (ValueError, IndexError):
            limit = 5
            print(f"\n🧪 TEST MODE: Processing first {limit} videos only")
    
    processor = BatchProcessor(verbose=True)
    success = processor.run(limit=limit)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
