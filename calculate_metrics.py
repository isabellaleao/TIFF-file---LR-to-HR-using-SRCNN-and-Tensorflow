#!/usr/bin/env python3
"""
Script para calcular métricas de qualidade da super-resolução:
MSE, MAE, PSNR e SSIM comparando ANADEM original vs resultado híbrido.
"""

import numpy as np
import cv2
from osgeo import gdal
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def _load_raster_data(filepath):
    """Carrega dados de um arquivo raster e retorna o array e o valor NoData."""
    ds = gdal.Open(filepath)
    if ds is None:
        raise FileNotFoundError(f"Não foi possível abrir o arquivo: {filepath}")
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    ds = None
    return data, nodata

def load_geotiff_data(filepath, max_size=1000):
    """Carrega dados de um arquivo GeoTIFF."""
    try:
        ds = gdal.Open(filepath)
        if ds is None:
            print(f"❌ Não foi possível abrir: {filepath}")
            return None, None
        
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        nodata_value = band.GetNoDataValue()
        geotransform = ds.GetGeoTransform()
        
        # Criar máscara de dados válidos
        valid_mask = ~np.isnan(data) & ~np.isinf(data)
        if nodata_value is not None:
            valid_mask = valid_mask & (data != nodata_value)
        
        # Redimensionar se muito grande para comparação
        if data.shape[0] > max_size or data.shape[1] > max_size:
            scale = min(max_size / data.shape[0], max_size / data.shape[1])
            new_height = int(data.shape[0] * scale)
            new_width = int(data.shape[1] * scale)
            
            from scipy.ndimage import zoom
            data_resized = zoom(data, (scale, scale), order=1)
            valid_mask_resized = zoom(valid_mask.astype(float), (scale, scale), order=1) > 0.5
        else:
            data_resized = data
            valid_mask_resized = valid_mask
        
        # Aplicar máscara
        data_masked = data_resized.copy()
        data_masked[~valid_mask_resized] = np.nan
        
        ds = None
        return data_masked, valid_mask_resized
        
    except Exception as e:
        print(f"❌ Erro ao carregar {filepath}: {e}")
        return None, None

def calculate_mse(original, enhanced):
    """Calcula Mean Squared Error."""
    # Filtrar apenas dados válidos
    valid_mask = ~np.isnan(original) & ~np.isnan(enhanced) & ~np.isinf(original) & ~np.isinf(enhanced)
    
    if np.sum(valid_mask) == 0:
        return np.nan
    
    original_valid = original[valid_mask]
    enhanced_valid = enhanced[valid_mask]
    
    mse = np.mean((original_valid - enhanced_valid) ** 2)
    return mse

def calculate_mae(original, enhanced):
    """Calcula Mean Absolute Error."""
    # Filtrar apenas dados válidos
    valid_mask = ~np.isnan(original) & ~np.isnan(enhanced) & ~np.isinf(original) & ~np.isinf(enhanced)
    
    if np.sum(valid_mask) == 0:
        return np.nan
    
    original_valid = original[valid_mask]
    enhanced_valid = enhanced[valid_mask]
    
    mae = np.mean(np.abs(original_valid - enhanced_valid))
    return mae

def calculate_psnr(original, enhanced):
    """Calcula Peak Signal-to-Noise Ratio."""
    # Filtrar apenas dados válidos
    valid_mask = ~np.isnan(original) & ~np.isnan(enhanced) & ~np.isinf(original) & ~np.isinf(enhanced)
    
    if np.sum(valid_mask) == 0:
        return np.nan
    
    original_valid = original[valid_mask]
    enhanced_valid = enhanced[valid_mask]
    
    # Normalizar para 0-1 para cálculo do PSNR
    orig_min, orig_max = np.min(original_valid), np.max(original_valid)
    enh_min, enh_max = np.min(enhanced_valid), np.max(enhanced_valid)
    
    if orig_max - orig_min < 1e-6 or enh_max - enh_min < 1e-6:
        return np.nan
    
    orig_norm = (original_valid - orig_min) / (orig_max - orig_min)
    enh_norm = (enhanced_valid - enh_min) / (enh_max - enh_min)
    
    # Calcular PSNR
    mse = np.mean((orig_norm - enh_norm) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr_value = 20 * np.log10(1.0 / np.sqrt(mse))
    return psnr_value

def calculate_ssim(original, enhanced):
    """Calcula Structural Similarity Index."""
    # Filtrar apenas dados válidos
    valid_mask = ~np.isnan(original) & ~np.isnan(enhanced) & ~np.isinf(original) & ~np.isinf(enhanced)
    
    if np.sum(valid_mask) == 0:
        return np.nan
    
    # Criar arrays com dados válidos
    original_clean = original.copy()
    enhanced_clean = enhanced.copy()
    
    # Substituir valores inválidos por média
    orig_valid = original[valid_mask]
    enh_valid = enhanced[valid_mask]
    
    if len(orig_valid) == 0 or len(enh_valid) == 0:
        return np.nan
    
    original_clean[~valid_mask] = np.mean(orig_valid)
    enhanced_clean[~valid_mask] = np.mean(enh_valid)
    
    # Normalizar para 0-1
    orig_min, orig_max = np.min(original_clean), np.max(original_clean)
    enh_min, enh_max = np.min(enhanced_clean), np.max(enhanced_clean)
    
    if orig_max - orig_min < 1e-6 or enh_max - enh_min < 1e-6:
        return np.nan
    
    orig_norm = (original_clean - orig_min) / (orig_max - orig_min)
    enh_norm = (enhanced_clean - enh_min) / (enh_max - enh_min)
    
    # Calcular SSIM
    ssim_value = ssim(orig_norm, enh_norm, data_range=1.0)
    return ssim_value

def calculate_metrics():
    """Calcula todas as métricas de qualidade."""
    print("📊 CALCULANDO MÉTRICAS DE QUALIDADE")
    print("=" * 50)
    
    # Caminhos dos arquivos
    anadem_original = 'data/images/ANADEM_AricanduvaBufferUTM.tif'
    anadem_hybrid = 'geospatial_output/anadem_hybrid_super_resolution_3x.tif'
    
    # Verificar se arquivos existem
    if not os.path.exists(anadem_original):
        print(f"❌ Arquivo original não encontrado: {anadem_original}")
        return
    
    if not os.path.exists(anadem_hybrid):
        print(f"❌ Arquivo híbrido não encontrado: {anadem_hybrid}")
        return
    
    print("📁 Carregando ANADEM original...")
    original_data, original_mask = load_geotiff_data(anadem_original, max_size=800)
    
    if original_data is None:
        print("❌ Falha ao carregar ANADEM original")
        return
    
    print("📁 Carregando resultado híbrido...")
    hybrid_data, hybrid_mask = load_geotiff_data(anadem_hybrid, max_size=800)
    
    if hybrid_data is None:
        print("❌ Falha ao carregar resultado híbrido")
        return
    
    print(f"✅ ANADEM original: {original_data.shape[1]}×{original_data.shape[0]} pixels")
    print(f"✅ Resultado híbrido: {hybrid_data.shape[1]}×{hybrid_data.shape[0]} pixels")
    
    # Verificar se as dimensões são compatíveis
    if original_data.shape != hybrid_data.shape:
        print("⚠️  Dimensões diferentes - redimensionando para comparação...")
        # Redimensionar o resultado híbrido para o tamanho do original
        hybrid_data = cv2.resize(hybrid_data, (original_data.shape[1], original_data.shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
    
    print("\n🔍 Calculando métricas...")
    
    # Calcular métricas
    mse = calculate_mse(original_data, hybrid_data)
    mae = calculate_mae(original_data, hybrid_data)
    psnr_value = calculate_psnr(original_data, hybrid_data)
    ssim_value = calculate_ssim(original_data, hybrid_data)
    
    # Exibir resultados
    print("\n📈 RESULTADOS DAS MÉTRICAS")
    print("=" * 40)
    print(f"MSE (Mean Squared Error):     {mse:.6f}")
    print(f"MAE (Mean Absolute Error):    {mae:.6f}")
    print(f"PSNR (Peak SNR):              {psnr_value:.2f} dB")
    print(f"SSIM (Structural Similarity): {ssim_value:.4f}")
    
    # Interpretação das métricas
    print("\n📊 INTERPRETAÇÃO DAS MÉTRICAS")
    print("=" * 40)
    
    if not np.isnan(mse):
        if mse < 0.001:
            print("✅ MSE: Excelente (muito baixo erro)")
        elif mse < 0.01:
            print("✅ MSE: Bom (baixo erro)")
        elif mse < 0.1:
            print("⚠️  MSE: Moderado (erro médio)")
        else:
            print("❌ MSE: Alto (erro elevado)")
    
    if not np.isnan(mae):
        if mae < 0.01:
            print("✅ MAE: Excelente (muito baixo erro absoluto)")
        elif mae < 0.1:
            print("✅ MAE: Bom (baixo erro absoluto)")
        elif mae < 1.0:
            print("⚠️  MAE: Moderado (erro absoluto médio)")
        else:
            print("❌ MAE: Alto (erro absoluto elevado)")
    
    if not np.isnan(psnr_value):
        if psnr_value > 40:
            print("✅ PSNR: Excelente (>40 dB)")
        elif psnr_value > 30:
            print("✅ PSNR: Bom (30-40 dB)")
        elif psnr_value > 20:
            print("⚠️  PSNR: Moderado (20-30 dB)")
        else:
            print("❌ PSNR: Baixo (<20 dB)")
    
    if not np.isnan(ssim_value):
        if ssim_value > 0.9:
            print("✅ SSIM: Excelente (>0.9)")
        elif ssim_value > 0.8:
            print("✅ SSIM: Bom (0.8-0.9)")
        elif ssim_value > 0.6:
            print("⚠️  SSIM: Moderado (0.6-0.8)")
        else:
            print("❌ SSIM: Baixo (<0.6)")
    
    # Salvar métricas em arquivo
    metrics_file = 'geospatial_output/metrics_results.txt'
    with open(metrics_file, 'w') as f:
        f.write("MÉTRICAS DE QUALIDADE - SUPER-RESOLUÇÃO HÍBRIDA\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Arquivo original: {anadem_original}\n")
        f.write(f"Arquivo híbrido: {anadem_hybrid}\n")
        f.write(f"Data: {np.datetime64('now')}\n\n")
        f.write("MÉTRICAS CALCULADAS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"MSE:  {mse:.6f}\n")
        f.write(f"MAE:  {mae:.6f}\n")
        f.write(f"PSNR: {psnr_value:.2f} dB\n")
        f.write(f"SSIM: {ssim_value:.4f}\n")
    
    print(f"\n💾 Métricas salvas em: {metrics_file}")
    
    # Estatísticas dos dados
    print("\n📊 ESTATÍSTICAS DOS DADOS")
    print("=" * 40)
    
    orig_valid = original_data[~np.isnan(original_data)]
    hybrid_valid = hybrid_data[~np.isnan(hybrid_data)]
    
    if len(orig_valid) > 0:
        print(f"ANADEM original:")
        print(f"  Range: {np.min(orig_valid):.2f} - {np.max(orig_valid):.2f}")
        print(f"  Média: {np.mean(orig_valid):.2f}")
        print(f"  Std:   {np.std(orig_valid):.2f}")
    
    if len(hybrid_valid) > 0:
        print(f"Resultado híbrido:")
        print(f"  Range: {np.min(hybrid_valid):.2f} - {np.max(hybrid_valid):.2f}")
        print(f"  Média: {np.mean(hybrid_valid):.2f}")
        print(f"  Std:   {np.std(hybrid_valid):.2f}")

def calculate_fair_metrics():
    """Calcula métricas justas comparando dados na mesma resolução."""
    
    print("\n" + "="*60)
    print("📊 MÉTRICAS JUSTAS - MESMA RESOLUÇÃO ESPACIAL")
    print("="*60)
    
    try:
        # 1. Carregar ANADEM original
        print("📁 Carregando ANADEM original...")
        anadem_data, anadem_nodata = _load_raster_data('data/images/ANADEM_AricanduvaBufferUTM.tif')
        print(f"✅ ANADEM: {anadem_data.shape[1]}×{anadem_data.shape[0]} pixels (30m/pixel)")
        
        # 2. Carregar resultado híbrido
        print("📁 Carregando resultado híbrido...")
        hybrid_data, hybrid_nodata = _load_raster_data('geospatial_output/anadem_hybrid_super_resolution_3x.tif')
        print(f"✅ Híbrido: {hybrid_data.shape[1]}×{hybrid_data.shape[0]} pixels (10m/pixel)")
        
        # 3. Upscaling do ANADEM para 10m (baseline bilinear)
        print("📐 Criando baseline bilinear do ANADEM...")
        anadem_upscaled = cv2.resize(anadem_data, (hybrid_data.shape[1], hybrid_data.shape[0]), 
                                    interpolation=cv2.INTER_LINEAR)
        print(f"✅ ANADEM upscaled: {anadem_upscaled.shape[1]}×{anadem_upscaled.shape[0]} pixels")
        
        # 4. Carregar GEOSAMPA e extrair região correspondente
        print("📁 Carregando GEOSAMPA...")
        geosampa_data, geosampa_nodata = _load_raster_data('data/images/MDTGeosampa_AricanduvaBufferUTM.tif')
        
        # Extrair região do GEOSAMPA (coordenadas conhecidas)
        x_start, y_start = 18437, 13802
        geosampa_region = geosampa_data[y_start:y_start+anadem_data.shape[0], 
                                       x_start:x_start+anadem_data.shape[1]]
        
        # Downscale GEOSAMPA para 10m/pixel (de 0.5m para 10m = 20x downscale)
        print("📐 Downscaling GEOSAMPA para 10m/pixel...")
        geosampa_downscaled = cv2.resize(geosampa_region, (hybrid_data.shape[1], hybrid_data.shape[0]), 
                                        interpolation=cv2.INTER_AREA)
        print(f"✅ GEOSAMPA downscaled: {geosampa_downscaled.shape[1]}×{geosampa_downscaled.shape[0]} pixels")
        
        # 5. Comparação 1: ANADEM Upscaled vs Resultado Híbrido
        print("\n" + "="*50)
        print("📊 COMPARAÇÃO 1: ANADEM Upscaled vs Resultado Híbrido")
        print("="*50)
        print("(Baseline bilinear vs Modelo + Interpolação)")
        
        metrics1 = calculate_metrics_pair(anadem_upscaled, hybrid_data, "ANADEM Upscaled", "Resultado Híbrido")
        
        # 6. Comparação 2: GEOSAMPA Downscaled vs Resultado Híbrido  
        print("\n" + "="*50)
        print("📊 COMPARAÇÃO 2: GEOSAMPA Downscaled vs Resultado Híbrido")
        print("="*50)
        print("(Referência de alta qualidade vs Modelo)")
        
        metrics2 = calculate_metrics_pair(geosampa_downscaled, hybrid_data, "GEOSAMPA Downscaled", "Resultado Híbrido")
        
        # 7. Comparação 3: ANADEM Upscaled vs GEOSAMPA Downscaled
        print("\n" + "="*50)
        print("📊 COMPARAÇÃO 3: ANADEM Upscaled vs GEOSAMPA Downscaled")
        print("="*50)
        print("(Baseline vs Referência de qualidade)")
        
        metrics3 = calculate_metrics_pair(anadem_upscaled, geosampa_downscaled, "ANADEM Upscaled", "GEOSAMPA Downscaled")
        
        # 8. Salvar todas as métricas
        print("\n💾 Salvando métricas...")
        with open('geospatial_output/metrics_fair_comparison.txt', 'w') as f:
            f.write("MÉTRICAS JUSTAS - MESMA RESOLUÇÃO ESPACIAL (10m/pixel)\n")
            f.write("="*60 + "\n\n")
            
            f.write("COMPARAÇÃO 1: ANADEM Upscaled vs Resultado Híbrido\n")
            f.write("-" * 50 + "\n")
            f.write(f"MSE: {metrics1['mse']:.4f}\n")
            f.write(f"MAE: {metrics1['mae']:.4f}\n")
            f.write(f"PSNR: {metrics1['psnr']:.2f} dB\n")
            f.write(f"SSIM: {metrics1['ssim']:.4f}\n\n")
            
            f.write("COMPARAÇÃO 2: GEOSAMPA Downscaled vs Resultado Híbrido\n")
            f.write("-" * 50 + "\n")
            f.write(f"MSE: {metrics2['mse']:.4f}\n")
            f.write(f"MAE: {metrics2['mae']:.4f}\n")
            f.write(f"PSNR: {metrics2['psnr']:.2f} dB\n")
            f.write(f"SSIM: {metrics2['ssim']:.4f}\n\n")
            
            f.write("COMPARAÇÃO 3: ANADEM Upscaled vs GEOSAMPA Downscaled\n")
            f.write("-" * 50 + "\n")
            f.write(f"MSE: {metrics3['mse']:.4f}\n")
            f.write(f"MAE: {metrics3['mae']:.4f}\n")
            f.write(f"PSNR: {metrics3['psnr']:.2f} dB\n")
            f.write(f"SSIM: {metrics3['ssim']:.4f}\n")
        
        print("✅ Métricas salvas em: geospatial_output/metrics_fair_comparison.txt")
        
    except Exception as e:
        print(f"❌ Erro ao calcular métricas justas: {e}")
        import traceback
        traceback.print_exc()

def calculate_metrics_pair(data1, data2, name1, name2):
    """Calcula métricas entre dois arrays de dados."""
    
    # Alinhar tamanhos se necessário
    min_rows = min(data1.shape[0], data2.shape[0])
    min_cols = min(data1.shape[1], data2.shape[1])
    
    data1 = data1[:min_rows, :min_cols]
    data2 = data2[:min_rows, :min_cols]
    
    # Criar máscaras válidas
    valid_mask1 = ~np.isnan(data1) & ~np.isinf(data1)
    valid_mask2 = ~np.isnan(data2) & ~np.isinf(data2)
    common_valid_mask = valid_mask1 & valid_mask2
    
    if np.sum(common_valid_mask) == 0:
        print(f"❌ Não há pixels válidos em comum entre {name1} e {name2}")
        return {'mse': np.nan, 'mae': np.nan, 'psnr': np.nan, 'ssim': np.nan}
    
    data1_valid = data1[common_valid_mask]
    data2_valid = data2[common_valid_mask]
    
    # Verificar se há dados válidos suficientes
    if len(data1_valid) < 10 or len(data2_valid) < 10:
        print(f"❌ Dados insuficientes para cálculo: {len(data1_valid)} pixels válidos")
        return {'mse': np.nan, 'mae': np.nan, 'psnr': np.nan, 'ssim': np.nan}
    
    # Verificar se há valores infinitos ou extremos
    if np.any(np.isinf(data1_valid)) or np.any(np.isinf(data2_valid)):
        print("⚠️  Valores infinitos detectados, filtrando...")
        finite_mask = ~np.isinf(data1_valid) & ~np.isinf(data2_valid)
        data1_valid = data1_valid[finite_mask]
        data2_valid = data2_valid[finite_mask]
        
        if len(data1_valid) < 10:
            print("❌ Dados insuficientes após filtro de infinitos")
            return {'mse': np.nan, 'mae': np.nan, 'psnr': np.nan, 'ssim': np.nan}
    
    # Verificar se há valores extremos (outliers)
    q1_1, q99_1 = np.percentile(data1_valid, [1, 99])
    q1_2, q99_2 = np.percentile(data2_valid, [1, 99])
    
    # Filtrar outliers extremos
    outlier_mask = (
        (data1_valid >= q1_1) & (data1_valid <= q99_1) &
        (data2_valid >= q1_2) & (data2_valid <= q99_2)
    )
    
    if np.sum(outlier_mask) > 100:  # Manter pelo menos 100 pixels
        data1_valid = data1_valid[outlier_mask]
        data2_valid = data2_valid[outlier_mask]
        print(f"📊 Filtrados outliers: {np.sum(outlier_mask)} pixels válidos")
    
    # Calcular métricas
    mse = mean_squared_error(data1_valid, data2_valid)
    mae = np.mean(np.abs(data1_valid - data2_valid))
    
    # PSNR
    global_min = min(np.min(data1_valid), np.min(data2_valid))
    global_max = max(np.max(data1_valid), np.max(data2_valid))
    data_range = global_max - global_min if global_max - global_min > 1e-6 else 1.0
    psnr_value = psnr(data1_valid, data2_valid, data_range=data_range)
    
    # SSIM em sub-região
    ssim_val = np.nan
    try:
        sub_size = 100
        for r in range(0, data1.shape[0] - sub_size, sub_size // 2):
            for c in range(0, data1.shape[1] - sub_size, sub_size // 2):
                sub1 = data1[r:r+sub_size, c:c+sub_size]
                sub2 = data2[r:r+sub_size, c:c+sub_size]
                sub_mask = common_valid_mask[r:r+sub_size, c:c+sub_size]
                
                if np.sum(sub_mask) > (sub_size * sub_size * 0.9):
                    sub_min = min(np.min(sub1[sub_mask]), np.min(sub2[sub_mask]))
                    sub_max = max(np.max(sub1[sub_mask]), np.max(sub2[sub_mask]))
                    
                    if sub_max - sub_min > 1e-6:
                        norm_sub1 = (sub1 - sub_min) / (sub_max - sub_min)
                        norm_sub2 = (sub2 - sub_min) / (sub_max - sub_min)
                        ssim_val = ssim(norm_sub1, norm_sub2, data_range=1.0)
                        break
            if not np.isnan(ssim_val):
                break
    except:
        pass
    
    # Exibir resultados
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    
    # Interpretação
    print(f"\n📊 INTERPRETAÇÃO:")
    if mse < 100:
        print("✅ MSE: Baixo (boa similaridade)")
    elif mse < 1000:
        print("⚠️  MSE: Moderado")
    else:
        print("❌ MSE: Alto (diferenças significativas)")
    
    if mae < 5:
        print("✅ MAE: Baixo (boa concordância)")
    elif mae < 10:
        print("⚠️  MAE: Moderado")
    else:
        print("❌ MAE: Alto (erro absoluto elevado)")
    
    if psnr_value > 30:
        print("✅ PSNR: Bom (>30 dB)")
    elif psnr_value > 20:
        print("⚠️  PSNR: Moderado (20-30 dB)")
    else:
        print("❌ PSNR: Baixo (<20 dB)")
    
    if ssim_val > 0.8:
        print("✅ SSIM: Alto (boa similaridade estrutural)")
    elif ssim_val > 0.6:
        print("⚠️  SSIM: Moderado (0.6-0.8)")
    else:
        print("❌ SSIM: Baixo (<0.6)")
    
    return {'mse': mse, 'mae': mae, 'psnr': psnr_value, 'ssim': ssim_val}

def main():
    """Função principal."""
    try:
        # Métricas originais (para comparação)
        print("🔍 Calculando métricas originais...")
        calculate_metrics()
        
        # Métricas justas (recomendadas)
        calculate_fair_metrics()
        
    except Exception as e:
        print(f"❌ Erro durante cálculo das métricas: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
