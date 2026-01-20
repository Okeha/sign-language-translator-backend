"""
Video Inventory Script for Motion Capture Pipeline

This script processes the wlasl_validated.json file to create a motion capture dataset
with one video per unique gloss (sign language word).

Output:
    motion_dataset.json - One video per unique gloss, ready for motion extraction
"""

import json
from pathlib import Path
from typing import Dict, List


class VideoInventory:
    def __init__(self, verbose: bool = True):
        """
        Initialize the Video Inventory system.
        
        Args:
            verbose: Enable detailed logging output
        """
        self.verbose = verbose
        self.base_path = Path(__file__).parent.parent / "model" / "finetune"
        self.validated_path = self.base_path / "data_engineering" / "datasets" / "wlasl_validated.json"
        self.output_path = Path(__file__).parent / "motion_dataset.json"
        
        self.validated_data: List[Dict] = []
        self.gloss_to_videos: Dict[str, List[Dict]] = {}
        self.motion_dataset: List[Dict] = []
            
    def load_validated_dataset(self):
        """Load the validated WLASL dataset (already filtered and validated)."""
        if self.verbose:
            print("\n📚 Loading validated WLASL dataset...")
        
        if not self.validated_path.exists():
            raise FileNotFoundError(f"Validated dataset not found: {self.validated_path}")
        
        with open(self.validated_path, 'r', encoding='utf-8') as f:
            self.validated_data = json.load(f)
        
        if self.verbose:
            print(f"   Loaded {len(self.validated_data)} validated video entries")
            
    def group_by_gloss(self):
        """Group videos by gloss."""
        if self.verbose:
            print("\n🔗 Grouping videos by gloss...")
        
        for entry in self.validated_data:
            gloss = entry.get('gloss', '').lower()
            video_path = entry.get('video_path', '')
            
            if gloss not in self.gloss_to_videos:
                self.gloss_to_videos[gloss] = []
            
            self.gloss_to_videos[gloss].append({
                'gloss': gloss,
                'video_path': video_path,
                'prompt': entry.get('prompt', '')
            })
        
        if self.verbose:
            print(f"   Found {len(self.gloss_to_videos)} unique glosses")
            
    def select_one_video_per_gloss(self):
        """Select the first available video for each unique gloss."""
        if self.verbose:
            print("\n✂️  Selecting one video per gloss...")
        
        for gloss in sorted(self.gloss_to_videos.keys()):
            videos = self.gloss_to_videos[gloss]
            # Select first video (deterministic)
            selected_video = videos[0]
            
            self.motion_dataset.append({
                'gloss': gloss,
                'video_path': selected_video['video_path'],
                'total_videos_available': len(videos)
            })
        
        if self.verbose:
            print(f"   Created motion dataset with {len(self.motion_dataset)} glosses")
            
    def save_motion_dataset(self):
        """Save the motion dataset to JSON file."""
        if self.verbose:
            print(f"\n💾 Saving motion dataset to {self.output_path}...")
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.motion_dataset, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"   ✅ Saved {len(self.motion_dataset)} gloss entries")
            
    def print_summary(self):
        """Print summary statistics."""
        if self.verbose:
            print("\n" + "="*60)
            print("📊 VIDEO INVENTORY SUMMARY")
            print("="*60)
            print(f"Total validated videos:         {len(self.validated_data)}")
            print(f"Unique glosses:                 {len(self.gloss_to_videos)}")
            print(f"Motion dataset entries:         {len(self.motion_dataset)}")
            print("="*60)
            
            # Show sample glosses
            print("\n📝 Sample glosses in motion dataset:")
            for entry in self.motion_dataset[:10]:
                print(f"   - {entry['gloss'].upper():15} → {entry['video_path']} ({entry['total_videos_available']} videos)")
            
            if len(self.motion_dataset) > 10:
                print(f"   ... and {len(self.motion_dataset) - 10} more")
            
            print(f"\n✅ Motion dataset saved to: {self.output_path}")
            
    def run(self):
        """Execute the complete video inventory pipeline."""
        print("\n" + "="*60)
        print("🎬 VIDEO INVENTORY PIPELINE")
        print("="*60)
        
        try:
            self.load_validated_dataset()
            self.group_by_gloss()
            self.select_one_video_per_gloss()
            self.save_motion_dataset()
            self.print_summary()
            
            print("\n✅ Video inventory completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Error during video inventory: {e}")
            raise


def main():
    """Main entry point for the video inventory script."""
    inventory = VideoInventory(verbose=True)
    inventory.run()


if __name__ == "__main__":
    main()
