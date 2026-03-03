import json
import random
from pathlib import Path
from collections import Counter

import pandas as pd
import sys
sys.path.append('./')

from src.utils import get_hash

def extract_sample_from_math(df, samples_per_level=40):
    """Extract samples uniformly from each level and subject."""
    extracted_samples = []
    levels = sorted(df['level'].unique())
    subjects = sorted(df['subject'].unique())

    # Get availability counts for each level-subject combination
    availability = {
        (level, subject): len(df[(df['level'] == level) & (df['subject'] == subject)])
        for level in levels for subject in subjects
    }

    # Calculate target distribution per level-subject
    base_count = samples_per_level // len(subjects)
    remainder = samples_per_level % len(subjects)

    target_per_level_subject = {}
    for level in levels:
        # Sort subjects by availability for this level
        level_subjects = sorted(subjects, key=lambda s: availability.get((level, s), 0), reverse=True)

        # Assign base count to subjects with enough samples
        assigned = {s: base_count for s in level_subjects if availability.get((level, s), 0) >= base_count}

        # Distribute remainder to subjects with most availability
        for subject in level_subjects[:remainder]:
            if subject in assigned and availability.get((level, subject), 0) > base_count:
                assigned[subject] += 1

        # Redistribute if needed
        total_assigned = sum(assigned.values())
        if total_assigned < samples_per_level:
            shortage = samples_per_level - total_assigned
            for subject in level_subjects:
                if shortage <= 0:
                    break
                current = assigned.get(subject, 0)
                max_extra = min(availability.get((level, subject), 0) - current, shortage)
                if max_extra > 0:
                    assigned[subject] = current + max_extra
                    shortage -= max_extra

        # Update target counts
        for subject in subjects:
            target_per_level_subject[(level, subject)] = assigned.get(subject, 0)

    # Extract samples according to distribution
    for (level, subject), target_count in target_per_level_subject.items():
        if target_count > 0:
            subset = df[(df['level'] == level) & (df['subject'] == subject)]
            sampled = subset.sample(n=target_count, random_state=42) if len(subset) >= target_count else subset

            for _, row in sampled.iterrows():
                extracted_samples.append({
                    '__id__': get_hash(row['problem']),
                    'level': f"Level {level}",
                    'type': subject,
                    'problem': row['problem'],
                    'final_answer': row['answer'],
                    'solution': row['solution'],
                    'solution_answer': row['answer']
                })

    return extracted_samples


def create_seed_pairs(extracted_samples):
    """Create all possible permutations of seed pairs from extracted samples."""
    return [
        {
            '__id__': f"{data1['__id__']}_{data2['__id__']}",
            'data1': data1,
            'data2': data2
        }
        for i, data1 in enumerate(extracted_samples)
        for j, data2 in enumerate(extracted_samples)
        if i != j
    ]


def print_distribution_stats(samples, title):
    """Print distribution statistics for extracted samples."""
    print(f"\n--- {title} ---")

    # Count distributions
    level_counts = Counter(s['level'] for s in samples)
    type_counts = Counter(s['type'] for s in samples)
    level_type_counts = Counter((s['level'], s['type']) for s in samples)

    print("\nLevel distribution:")
    for level in sorted(level_counts):
        print(f"  {level}: {level_counts[level]}")

    print("\nType distribution:")
    for type_ in sorted(type_counts):
        print(f"  {type_}: {type_counts[type_]}")

    print("\nLevel x Type distribution:")
    for level in ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']:
        for type_ in sorted({t[1] for t in level_type_counts if t[0] == level}):
            count = level_type_counts.get((level, type_), 0)
            print(f"  {level} - {type_}: {count}")


def main(samples_per_level=40):
    """Generate seed pairs from MATH dataset."""
    print("Loading MATH dataset...")
    df = pd.read_parquet('./data/MATH/train-00000-of-00001.parquet')
    print(f"Total samples: {len(df)}")

    # Dataset info
    levels = sorted(df['level'].unique())
    print(f"\nLevels: {levels}")
    print(f"Subjects: {df['subject'].unique()}")
    print("\nLevel distribution:")
    print(df['level'].value_counts().sort_index())

    total_seeds = len(levels) * samples_per_level
    print(f"\nTotal seeds to extract: {total_seeds} ({samples_per_level} per level)")

    # Extract samples
    extracted_samples = extract_sample_from_math(df, samples_per_level)
    print(f"Extracted {len(extracted_samples)} samples")

    print_distribution_stats(extracted_samples, "Distribution of Extracted Samples")

    # Save individual seeds
    seed_file = Path(f'./data/MATH_seeds_{len(extracted_samples)}.jsonl')
    seed_file.parent.mkdir(exist_ok=True)
    with seed_file.open('w', encoding='utf-8') as f:
        for seed in extracted_samples:
            f.write(json.dumps(seed, ensure_ascii=False) + '\n')
    print(f"\nSaved individual seeds to: {seed_file}")

    # Create pairs
    seed_pairs = create_seed_pairs(extracted_samples)
    total_pairs = len(extracted_samples) * (len(extracted_samples) - 1)
    print(f"\nCreated {len(seed_pairs)} pairs ({total_pairs} total permutations)")

    # Save pairs
    output_file = Path(f'./data/MATH_full_permutations_{len(extracted_samples)}*{len(extracted_samples) - 1}.jsonl')
    output_file.parent.mkdir(exist_ok=True)

    with output_file.open('w', encoding='utf-8') as f:
        for pair in seed_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"\nSaved pairs to: {output_file}")

    # Quick verification
    with seed_file.open('r', encoding='utf-8') as f:
        seed_sample = json.loads(f.readline())
        print("\nSeed format verified:")
        print(f"  Keys: {list(seed_sample.keys())}")
        print(f"  ID: {seed_sample['__id__'][:16]}...")
        print(f"  Level: {seed_sample['level']}")
        print(f"  Type: {seed_sample['type']}")

    with output_file.open('r', encoding='utf-8') as f:
        pair_sample = json.loads(f.readline())
        print("\nPair format verified:")
        print(f"  Keys: {list(pair_sample.keys())}")
        print(f"  data1 Level: {pair_sample['data1']['level']}")
        print(f"  data2 Level: {pair_sample['data2']['level']}")

    return seed_pairs


if __name__ == '__main__':
    random.seed(42)
    samples_per_level = 40
    seed_pairs = main(samples_per_level=samples_per_level)
    total_seeds = len(set(p['data1']['__id__'] for p in seed_pairs))
    print(f"\nGenerated {len(seed_pairs)} seed pairs from {total_seeds} unique seeds!")
    print(f"Total permutations: {total_seeds} × ({total_seeds} - 1) = {total_seeds * (total_seeds - 1)}")