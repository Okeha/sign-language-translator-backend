"""
Video Dataset Validator
Removes corrupted videos from the dataset before training.
"""
import json
import decord
from pathlib import Path
from tqdm import tqdm

def validate_dataset(input_json="datasets/wlasl_cleaned.json", output_json="datasets/wlasl_validated.json"):
    """
    Validate all videos in the dataset and create a cleaned version.
    
    Args:
        input_json: Path to the input dataset JSON
        output_json: Path to save the validated dataset
    """
    print(f"🔍 Loading dataset from: {input_json}")
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Total samples in dataset: {len(data)}")
    
    valid_samples = []
    corrupted_videos = []
    
    print("\n🔄 Validating videos...")
    for sample in tqdm(data, desc="Validating"):
        video_path = f"{sample['video_path']}"
        
        try:
            # Try to open and read the video
            vr = decord.VideoReader(video_path)
            
            # Check if video has frames
            if len(vr) > 0:
                valid_samples.append(sample)
            else:
                corrupted_videos.append({
                    'video_path': video_path,
                    'gloss': sample.get('gloss', 'unknown'),
                    'reason': 'Empty video (0 frames)'
                })
                
        except Exception as e:
            corrupted_videos.append({
                'video_path': video_path,
                'gloss': sample.get('gloss', 'unknown'),
                'reason': str(e)
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Validation Summary")
    print(f"{'='*60}")
    print(f"✅ Valid videos:     {len(valid_samples)} ({100*len(valid_samples)/len(data):.1f}%)")
    print(f"❌ Corrupted videos: {len(corrupted_videos)} ({100*len(corrupted_videos)/len(data):.1f}%)")
    print(f"{'='*60}\n")
    
    # Save validated dataset
    if len(valid_samples) > 0:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(valid_samples, f, indent=2)
        
        print(f"💾 Saved validated dataset to: {output_path.absolute()}")
        print(f"   {len(valid_samples)} valid samples\n")
    else:
        print("⚠️  No valid samples found! Cannot save dataset.\n")
        return
    
    # Save corrupted videos log
    if len(corrupted_videos) > 0:
        corrupted_log = output_path.parent / "corrupted_videos.json"
        with open(corrupted_log, 'w') as f:
            json.dump(corrupted_videos, f, indent=2)
        
        print(f"📝 Saved corrupted videos log to: {corrupted_log.absolute()}")
        print(f"   {len(corrupted_videos)} corrupted videos\n")
        
        # Print first 10 corrupted videos
        print("❌ Sample of corrupted videos:")
        for i, cv in enumerate(corrupted_videos[:10], 1):
            print(f"   {i}. {cv['video_path']} ({cv['gloss']})")
            print(f"      Reason: {cv['reason']}")
        
        if len(corrupted_videos) > 10:
            print(f"   ... and {len(corrupted_videos) - 10} more (see corrupted_videos.json)")
    
    print(f"\n{'='*60}")
    print("✅ Validation complete!")
    print(f"{'='*60}")
    print(f"\n💡 Next steps:")
    print(f"   1. Review corrupted_videos.json")
    print(f"   2. Re-download or fix corrupted videos if needed")
    print(f"   3. Update your training script to use: {output_json}")
    print()

if __name__ == "__main__":
    validate_dataset(
        input_json="datasets/wlasl_cleaned.json",
        output_json="datasets/wlasl_validated.json"
    )
