#!/usr/bin/env python3
import os
import numpy as np
import argparse
from tqdm import tqdm
from libs.audio import WaveReader

def preprocess(mix_scp, ref_scp, aux_scp, output_dir, chunk_size=80000, least=40000, sample_rate=8000):
    """
    Decodes, chunks, pads, and saves audio data offline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize readers to decode audio files
    mix_reader = WaveReader(mix_scp, sample_rate=sample_rate)
    ref_reader = WaveReader(ref_scp, sample_rate=sample_rate)
    aux_reader = WaveReader(aux_scp, sample_rate=sample_rate)
    
    # New scp files to store paths to the offline chunks
    new_mix_scp = open(os.path.join(output_dir, "mix_chunks.scp"), "w")
    new_ref_scp = open(os.path.join(output_dir, "ref_chunks.scp"), "w")
    new_aux_scp = open(os.path.join(output_dir, "aux_chunks.scp"), "w")
    
    print(f"Processing {len(mix_reader)} utterances...")
    
    chunk_counter = 0
    for key in tqdm(mix_reader.index_keys):
        # 1. Waveform Decoding (Happens once here instead of every epoch)
        mix = mix_reader[key].astype(np.float32)
        ref = ref_reader[key].astype(np.float32)
        aux = aux_reader[key].astype(np.float32)
        
        N = mix.size
        if N < least:
            continue
            
        chunks = []
        
        # 2. Padding Short Audio Offline
        if N < chunk_size:
            P = chunk_size - N
            mix_pad = np.pad(mix, (0, P), "constant")
            ref_pad = np.pad(ref, (0, P), "constant")
            chunks.append((mix_pad, ref_pad))
        else:
            # 3. Pre-Chunking Long Audio Offline
            s = 0
            while s + chunk_size <= N:
                mix_chunk = mix[s:s + chunk_size]
                ref_chunk = ref[s:s + chunk_size]
                chunks.append((mix_chunk, ref_chunk))
                s += least # Using step shift size (least) matching your ChunkSplitter logic
                
        # Save extracted chunks and reference enrollment to disk
        for idx, (m_chk, r_chk) in enumerate(chunks):
            chunk_id = f"chk_{chunk_counter:07d}"
            
            mix_path = os.path.join(output_dir, f"{chunk_id}_mix.npy")
            ref_path = os.path.join(output_dir, f"{chunk_id}_ref.npy")
            aux_path = os.path.join(output_dir, f"{chunk_id}_aux.npy")
            
            # Save raw binary arrays
            np.save(mix_path, m_chk)
            np.save(ref_path, r_chk)
            np.save(aux_path, aux) # Aux shape stays native, padded during runtime collate
            
            # Write to new manifest files
            new_mix_scp.write(f"{chunk_id} {mix_path}\n")
            new_ref_scp.write(f"{chunk_id} {ref_path}\n")
            new_aux_scp.write(f"{chunk_id} {aux_path}\n")
            
            chunk_counter += 1

    new_mix_scp.close()
    new_ref_scp.close()
    new_aux_scp.close()
    print(f"Done! Extracted and saved {chunk_counter} total uniform chunks to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Audio Preprocessing")
    parser.add_argument("--mix_scp", type=str, required=True)
    parser.add_argument("--ref_scp", type=str, required=True)
    parser.add_argument("--aux_scp", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    preprocess(args.mix_scp, args.ref_scp, args.aux_scp, args.out_dir)