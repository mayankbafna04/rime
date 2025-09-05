import os
import numpy as np
from tqdm import tqdm
from Bio import SeqIO


def voss_mapping(sequence):
    """Converts a DNA sequence into four binary indicator sequences (Voss mapping)[cite: 252]."""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0,0,0,0]}
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence.upper()]).T


def mgwt_feature_extraction(indicator_sequences):
    """
    Simplified Modified Gabor-Wavelet Transform (MGWT) to extract features[cite: 80].
    We use the power spectrum from a Fourier Transform to find the three-base periodicity (TBP)[cite: 23, 99].
    """
    total_spectrum = np.zeros(len(indicator_sequences[0]))
    for seq in indicator_sequences:
        fft_result = np.fft.fft(seq - np.mean(seq)) # Zero-mean
        power_spectrum = np.abs(fft_result)**2
        total_spectrum += power_spectrum
    return total_spectrum


def create_labeled_frames(sequence, exon_coords, frame_len=256, overlap=100):
    """
    Creates framed data from a DNA sequence and labels them as exon (1) or intron (0)[cite: 260].
    """
    frames, labels = [], []
    is_exon_array = np.zeros(len(sequence), dtype=int)
    for start, end in exon_coords:
        is_exon_array[start:end] = 1 # Use 0-based indexing

    step = frame_len - overlap
    for i in range(0, len(sequence) - frame_len + 1, step):
        frame_seq = sequence[i:i + frame_len]
        frame_labels = is_exon_array[i:i + frame_len]
        
        label = 1 if np.mean(frame_labels) > 0.5 else 0
        frames.append(frame_seq)
        labels.append(label)
    return frames, labels


def preprocess_genbank_data(directory_path):
    """
    Main preprocessing pipeline: Reads all GenBank files in a directory and prepares data for the CNN.
    """
    all_features, all_labels = [], []
    print(f"Scanning for GenBank files in '{directory_path}'...")
    gb_files = [f for f in os.listdir(directory_path) if f.endswith(('.gb', '.gbk'))]
    if len(gb_files) == 0:
        raise FileNotFoundError(
            f"No GenBank files found in {directory_path}. Expected files with extensions .gb or .gbk."
        )

    for filename in tqdm(gb_files, desc="Processing GenBank Files"):
        filepath = os.path.join(directory_path, filename)
        for record in SeqIO.parse(filepath, "genbank"):
            sequence = str(record.seq)
            exon_coords = []
            for feature in record.features:
                if feature.type == "CDS":
                    start = int(feature.location.start)
                    end = int(feature.location.end)
                    exon_coords.append((start, end))
            
            if not exon_coords: continue

            frames, labels = create_labeled_frames(sequence, exon_coords)
            
            for i in range(len(frames)):
                voss_seqs = voss_mapping(frames[i])
                spectrum = mgwt_feature_extraction(voss_seqs)
                
                # Normalize spectrum to [0, 1] range [cite: 258]
                if np.max(spectrum) > np.min(spectrum):
                    spectrum_norm = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
                else:
                    spectrum_norm = np.zeros_like(spectrum)

                all_features.append(spectrum_norm)
                all_labels.append(labels[i])
            
    # Reshape features into 16x16x1 matrices for the CNN [cite: 261]
    X = np.array(all_features).reshape(-1, 16, 16, 1)
    y = np.array(all_labels)
    
    print(f"\nPreprocessing complete. Generated {len(X)} samples.")
    print(f"Class distribution - 0 (Intron): {np.sum(y==0)}, 1 (Exon): {np.sum(y==1)}")
    return X, y