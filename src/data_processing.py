import os
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import pywt
from scipy import signal
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler


def z_curve_mapping(sequence):
    """
    Z-curve representation for capturing nucleotide composition patterns.
    More sophisticated than Voss mapping for capturing genomic patterns.
    """
    mapping = {
        'A': [1, 1, -1], 'T': [-1, -1, -1],
        'G': [1, -1, 1], 'C': [-1, 1, 1],
        'N': [0, 0, 0]
    }
    
    z_coords = []
    cumsum = np.array([0.0, 0.0, 0.0])
    
    for base in sequence.upper():
        cumsum += mapping.get(base, [0, 0, 0])
        z_coords.append(cumsum.copy())
    
    return np.array(z_coords).T


def tetrahedron_mapping(sequence):
    """
    Maps DNA sequences to 3D tetrahedron vertices for better geometric representation.
    """
    mapping = {
        'A': np.array([1, 1, 1]) / np.sqrt(3),
        'T': np.array([1, -1, -1]) / np.sqrt(3),
        'G': np.array([-1, 1, -1]) / np.sqrt(3),
        'C': np.array([-1, -1, 1]) / np.sqrt(3),
        'N': np.array([0, 0, 0])
    }
    
    coords = np.array([mapping.get(base, mapping['N']) for base in sequence.upper()])
    return coords.T


def compute_codon_usage_bias(sequence):
    """
    Computes codon usage bias features which are strong indicators of coding regions.
    """
    codon_freq = {}
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3].upper()
        if 'N' not in codon:
            codon_freq[codon] = codon_freq.get(codon, 0) + 1
    
    # Normalize frequencies
    total = sum(codon_freq.values()) + 1e-10
    codon_probs = np.array([codon_freq.get(codon, 0) / total for codon in codon_freq.keys()])
    
    # Compute entropy as measure of codon bias
    codon_entropy = entropy(codon_probs + 1e-10)
    
    # Create feature vector (64 possible codons)
    feature_vector = np.zeros(64)
    codon_to_idx = {}
    idx = 0
    for b1 in 'ACGT':
        for b2 in 'ACGT':
            for b3 in 'ACGT':
                codon = b1 + b2 + b3
                codon_to_idx[codon] = idx
                if codon in codon_freq:
                    feature_vector[idx] = codon_freq[codon] / total
                idx += 1
    
    return feature_vector, codon_entropy


def wavelet_transform_features(signal_data, wavelet='db4', level=4):
    """
    Multi-resolution wavelet decomposition for capturing both local and global patterns.
    """
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    features = []
    
    for i, coeff in enumerate(coeffs):
        # Statistical features from each decomposition level
        features.extend([
            np.mean(coeff),
            np.std(coeff),
            np.max(np.abs(coeff)),
            entropy(np.histogram(coeff, bins=10)[0] + 1e-10)
        ])
    
    return np.array(features)


def compute_spectral_features(sequence):
    """
    Advanced spectral analysis including three-base periodicity and spectral entropy.
    """
    # Multiple numerical representations
    voss_seqs = voss_mapping(sequence)
    z_curve = z_curve_mapping(sequence)
    
    features = []
    
    # Analyze each representation
    for seq in voss_seqs:
        if len(seq) > 0:
            # Power spectral density
            freqs, psd = signal.periodogram(seq - np.mean(seq))
            
            # Three-base periodicity strength (key exon indicator)
            tbp_freq = len(seq) / 3
            tbp_idx = np.argmin(np.abs(freqs - 1/3))
            tbp_power = psd[tbp_idx]
            
            # Spectral entropy
            psd_norm = psd / (np.sum(psd) + 1e-10)
            spec_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            
            # Peak frequency
            peak_freq = freqs[np.argmax(psd)]
            
            features.extend([tbp_power, spec_entropy, peak_freq])
    
    # Z-curve spectral features
    for z_comp in z_curve:
        if len(z_comp) > 0:
            freqs, psd = signal.periodogram(z_comp)
            features.append(np.max(psd))
            features.append(np.mean(psd))
    
    return np.array(features)


def compute_positional_nucleotide_features(sequence):
    """
    Position-specific nucleotide frequencies and dinucleotide frequencies.
    """
    features = []
    
    # Positional nucleotide frequencies (first, second, third codon positions)
    for pos in range(3):
        pos_seq = sequence[pos::3]
        for base in 'ACGT':
            features.append(pos_seq.count(base) / (len(pos_seq) + 1))
    
    # Dinucleotide frequencies
    dinuc_freq = {}
    for i in range(len(sequence) - 1):
        dinuc = sequence[i:i+2].upper()
        if 'N' not in dinuc:
            dinuc_freq[dinuc] = dinuc_freq.get(dinuc, 0) + 1
    
    total = sum(dinuc_freq.values()) + 1
    for b1 in 'ACGT':
        for b2 in 'ACGT':
            dinuc = b1 + b2
            features.append(dinuc_freq.get(dinuc, 0) / total)
    
    return np.array(features)


def voss_mapping(sequence):
    """Enhanced Voss mapping with normalization."""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 
               'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence.upper()]).T


def extract_comprehensive_features(sequence):
    """
    Extracts a comprehensive set of features combining multiple techniques.
    """
    all_features = []
    
    # 1. Spectral features (most important for exon detection)
    spectral_feats = compute_spectral_features(sequence)
    all_features.extend(spectral_feats)
    
    # 2. Codon usage features
    codon_feats, codon_ent = compute_codon_usage_bias(sequence)
    all_features.extend(codon_feats[:20])  # Top 20 codon frequencies
    all_features.append(codon_ent)
    
    # 3. Wavelet features from multiple representations
    voss_seqs = voss_mapping(sequence)
    for vs in voss_seqs[:2]:  # Use A and C channels
        wavelet_feats = wavelet_transform_features(vs)
        all_features.extend(wavelet_feats)
    
    # 4. Z-curve features
    z_curve = z_curve_mapping(sequence)
    for z_comp in z_curve:
        all_features.extend([
            np.mean(z_comp), np.std(z_comp),
            np.min(z_comp), np.max(z_comp)
        ])
    
    # 5. Positional and dinucleotide features
    pos_feats = compute_positional_nucleotide_features(sequence)
    all_features.extend(pos_feats)
    
    # 6. Sequence composition features
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    at_content = (sequence.count('A') + sequence.count('T')) / len(sequence)
    all_features.extend([gc_content, at_content])
    
    # 7. Open reading frame indicators
    start_codons = ['ATG']
    stop_codons = ['TAA', 'TAG', 'TGA']
    
    has_start = any(sequence[i:i+3] in start_codons for i in range(0, len(sequence)-2, 3))
    has_stop = any(sequence[i:i+3] in stop_codons for i in range(0, len(sequence)-2, 3))
    all_features.extend([float(has_start), float(has_stop)])
    
    return np.array(all_features)


def create_labeled_frames_enhanced(sequence, exon_coords, frame_len=300, overlap=150):
    """
    Enhanced frame creation with better overlap and frame size.
    Using 300bp frames to capture complete exon patterns.
    """
    frames, labels = [], []
    is_exon_array = np.zeros(len(sequence), dtype=int)
    
    for start, end in exon_coords:
        is_exon_array[start:end] = 1
    
    step = frame_len - overlap
    for i in range(0, len(sequence) - frame_len + 1, step):
        frame_seq = sequence[i:i + frame_len]
        frame_labels = is_exon_array[i:i + frame_len]
        
        # More sophisticated labeling: consider the proportion and position
        exon_ratio = np.mean(frame_labels)
        
        # Check if frame contains exon boundary (important feature)
        has_boundary = 0 < exon_ratio < 1
        
        # Label based on majority voting with boundary consideration
        if exon_ratio > 0.6:
            label = 1
        elif exon_ratio < 0.4:
            label = 0
        else:
            # For boundary regions, check central portion
            central_ratio = np.mean(frame_labels[frame_len//4:3*frame_len//4])
            label = 1 if central_ratio > 0.5 else 0
        
        frames.append(frame_seq)
        labels.append(label)
    
    return frames, labels


def preprocess_genbank_data_enhanced(directory_path, max_features=256):
    """
    Enhanced preprocessing pipeline with advanced feature extraction.
    """
    all_features, all_labels = [], []
    
    print(f"Scanning for GenBank files in '{directory_path}'...")
    gb_files = [f for f in os.listdir(directory_path) if f.endswith(('.gb', '.gbk'))]
    
    if len(gb_files) == 0:
        raise FileNotFoundError(
            f"No GenBank files found in {directory_path}. Expected files with extensions .gb or .gbk."
        )
    
    print("Extracting comprehensive genomic features...")
    
    for filename in tqdm(gb_files, desc="Processing GenBank Files"):
        filepath = os.path.join(directory_path, filename)
        
        for record in SeqIO.parse(filepath, "genbank"):
            sequence = str(record.seq)
            
            # Extract CDS (coding sequence) coordinates
            exon_coords = []
            for feature in record.features:
                if feature.type == "CDS":
                    start = int(feature.location.start)
                    end = int(feature.location.end)
                    exon_coords.append((start, end))
            
            if not exon_coords:
                continue
            
            # Create frames with enhanced parameters
            frames, labels = create_labeled_frames_enhanced(sequence, exon_coords)
            
            # Extract comprehensive features for each frame
            for frame_seq, label in zip(frames, labels):
                try:
                    features = extract_comprehensive_features(frame_seq)
                    all_features.append(features)
                    all_labels.append(label)
                except Exception as e:
                    continue  # Skip problematic frames
    
    if len(all_features) == 0:
        raise ValueError("No valid features extracted from the data")
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Feature selection/reduction to fixed size
    if X.shape[1] > max_features:
        # Use variance threshold to select most informative features
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-max_features:]
        X = X[:, top_indices]
    elif X.shape[1] < max_features:
        # Pad with zeros if needed
        padding = np.zeros((X.shape[0], max_features - X.shape[1]))
        X = np.hstack([X, padding])
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape for CNN (16x16x1 format)
    X = X.reshape(-1, 16, 16, 1)
    
    print(f"\nPreprocessing complete. Generated {len(X)} samples.")
    print(f"Class distribution - 0 (Intron): {np.sum(y==0)}, 1 (Exon): {np.sum(y==1)}")
    
    return X, y