import os
import glob
import argparse
import numpy as np
import torch
import torchaudio
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# Importamos os backends de extração de F0 que já existem no projeto
try:
    from extract_embs.extract_f0 import BACKEND_FN, device
except ImportError:
    print("Aviso: Não foi possível importar BACKEND_FN de extract_embs.extract_f0.")
    BACKEND_FN = {}


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 5) -> np.ndarray:
    """
    Aplica um filtro passa-baixa Butterworth (Zero-phase filtfilt).
    """
    nyq = 0.5 * fs  # Frequência de Nyquist
    normal_cutoff = cutoff / nyq
    
    # Previne erros caso a frequência de corte seja inválida para a taxa de amostragem
    if normal_cutoff >= 1.0:
        return data
    if normal_cutoff <= 0:
        return np.zeros_like(data)
        
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def process_audio(file_path: str, output_path: str, backend_name: str):
    """
    Carrega o áudio, estima o F0, calcula o cutoff e aplica o LPF.
    """
    # 1. Carrega o áudio
    audio_data, sr = torchaudio.load(file_path)
    
    # Converte estéreo para mono, se necessário
    if audio_data.dim() > 1 and audio_data.shape[0] > 1:
        audio_data = audio_data.mean(dim=0)
    # .copy() evita o erro "At least one stride in the given numpy array is negative"
    audio_np = audio_data.squeeze().numpy().copy()

    # 2. Estima a frequência fundamental (F0_contour)
    if backend_name in BACKEND_FN:
        extract_fn = BACKEND_FN[backend_name]
        # Usamos hop_length=160 para 16kHz (~10ms)
        f0_contour = extract_fn(audio_np, sr, hop_length=int(sr * 0.01))
    else:
        # Fallback usando librosa (pYIN) se o backend não for encontrado
        import librosa
        f0_contour, _, _ = librosa.pyin(audio_np, fmin=50.0, fmax=1100.0, sr=sr)
        f0_contour = np.nan_to_num(f0_contour, nan=0.0)

    # 3. Calcula F0 médio do segmento (ignorando frames não-vozeados onde f0 == 0)
    valid_f0 = f0_contour[f0_contour > 0]
    if len(valid_f0) > 0:
        avg_f0 = np.mean(valid_f0)
    else:
        # Fallback seguro se não for detectada voz
        avg_f0 = 150.0 

    # 4. Determina a frequência de corte dinamicamente
    # cutoff_freq = 420.2 * (1 - exp(-0.0124 * avg_F0))
    cutoff_freq = 420.2 * (1 - np.exp(-0.0124 * avg_f0))

    # Limita o cutoff para não ser excessivamente baixo (segurança DSP)
    cutoff_freq = max(50.0, cutoff_freq)

    # 5. Remove altas frequências (Aplica o Low-Pass Filter)
    filtered_audio = butter_lowpass_filter(audio_np, cutoff_freq, sr, order=5)

    # Compensa a perda de volume do filtro passa-baixa (Normalização de Pico)
    max_amp = np.max(np.abs(filtered_audio))
    if max_amp > 0:
        filtered_audio = (filtered_audio / max_amp) * 0.95 # Normaliza para 95% do volume máximo

    # 6. Salva o resultado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define caminhos para o original e o filtrado
    base, ext = os.path.splitext(output_path)
    path_filtered = f"{base}_filtered{ext}"
    path_original = f"{base}_original{ext}"

    # Salva o filtrado
    # .copy() aqui é CRUCIAL pois o filtfilt costuma retornar arrays que o torch não aceita diretamente
    filtered_tensor = torch.from_numpy(filtered_audio.copy()).unsqueeze(0).float()
    torchaudio.save(path_filtered, filtered_tensor, sr)

    # Salva o original (já convertido para mono) para comparação direta
    original_tensor = torch.from_numpy(audio_np.copy()).unsqueeze(0).float()
    torchaudio.save(path_original, original_tensor, sr)
    
    return avg_f0, cutoff_freq


def main():
    parser = argparse.ArgumentParser(description="Aplica LPF dinâmico baseado no F0 médio para isolar a prosódia.")
    parser.add_argument("-i", "--input-dir", 
                        default=r"F:\Git\Datasets\mos_last\Datasets_mos\bvcc\main\DATA\wav", 
                        help="Diretório contendo os áudios .wav de entrada (padrão: F:\\Git\\Datasets\\mos_last\\Datasets_mos\\bvcc\\main\\DATA\\wav)")
    parser.add_argument("-o", "--output-dir", 
                        default=r"F:\Git\Embeddings2\dataset_emb", 
                        help="Diretório onde os áudios filtrados serão salvos (padrão: F:\\Git\\Embeddings2\\dataset_emb)")
    parser.add_argument("-m", "--backend", default="crepe", choices=list(BACKEND_FN.keys()) if BACKEND_FN else ["crepe"], 
                        help="Backend para estimar F0 (padrão: crepe)")

    args = parser.parse_args()

    # Busca todos os arquivos .wav recursivamente
    filelist = glob.glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True)
    
    if not filelist:
        print(f"Nenhum arquivo .wav encontrado em '{args.input_dir}'")
        return

    # Limita para apenas 2 arquivos para teste rápido
    filelist = filelist[:2]

    print(f"Iniciando filtragem de prosódia em {len(filelist)} arquivos usando '{args.backend}' para F0...")

    stats = []
    for filepath in tqdm(filelist, desc="Filtrando"):
        # Preserva estrutura de subpastas
        rel_path = os.path.relpath(filepath, args.input_dir)
        output_filepath = os.path.join(args.output_dir, rel_path)
        
        try:
            avg_f0, cutoff = process_audio(filepath, output_filepath, args.backend)
            stats.append((avg_f0, cutoff))
        except Exception as e:
            print(f"\nErro ao processar {filepath}: {e}")

    if stats:
        avg_f0_all = np.mean([s[0] for s in stats])
        avg_cutoff_all = np.mean([s[1] for s in stats])
        print("\n=== Resumo do Processamento ===")
        print(f"F0 médio global detectado: {avg_f0_all:.2f} Hz")
        print(f"Frequência de corte média aplicada: {avg_cutoff_all:.2f} Hz")
        print("Concluído!")

if __name__ == "__main__":
    main()
