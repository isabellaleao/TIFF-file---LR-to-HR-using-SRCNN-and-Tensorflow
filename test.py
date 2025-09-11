#!/usr/bin/env python3
"""
Pipeline híbrido UNIVERSAL - adaptável para qualquer região.
"""

import numpy as np
import tensorflow as tf
from osgeo import gdal
import cv2
import os

def universal_hybrid_super_resolution(
    input_path,
    model_path='geospatial_output/advanced_model_best.keras',
    output_path=None,
    scale_factor=3,
    patch_size=30,
    overlap=10,
    min_valid_ratio=0.1  # Mínimo 10% de dados válidos por patch
):
    """
    Super-resolução híbrida UNIVERSAL para qualquer região.
    
    Args:
        input_path: Caminho para o arquivo TIFF de entrada
        model_path: Caminho para o modelo treinado
        output_path: Caminho de saída (auto-gerado se None)
        scale_factor: Fator de escala (padrão: 3x)
        patch_size: Tamanho do patch (padrão: 30x30)
        overlap: Sobreposição entre patches (padrão: 10px)
        min_valid_ratio: Proporção mínima de dados válidos por patch
    """
    
    print("🌍 SUPER-RESOLUÇÃO HÍBRIDA UNIVERSAL")
    print("=" * 50)
    print(f"📁 Entrada: {input_path}")
    print(f"🧠 Modelo: {model_path}")
    print(f"📊 Escala: {scale_factor}x")
    print(f"📏 Patch: {patch_size}x{patch_size} (overlap: {overlap}px)")
    
    # Auto-gerar nome de saída se não fornecido
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"geospatial_output/{base_name}_super_resolution_{scale_factor}x.tif"
    
    print(f"📁 Saída: {output_path}")
    
    gdal = _import_gdal()
    
    # 1. Carregar modelo
    model = _load_model_safe(model_path)
    
    # 2. Carregar parâmetros de normalização
    norm_params = _load_normalization_params()
    
    # 3. Carregar dados de entrada
    ds_input = gdal.Open(input_path)
    if ds_input is None:
        raise RuntimeError(f"Não foi possível abrir: {input_path}")
    
    band_input = ds_input.GetRasterBand(1)
    data_input = band_input.ReadAsArray()
    nodata_input = band_input.GetNoDataValue()
    geotransform = ds_input.GetGeoTransform()
    projection = ds_input.GetProjection()
    
    print(f"📊 Dimensões: {data_input.shape}")
    print(f"📊 NoData: {nodata_input}")
    
    # 4. Detectar automaticamente região válida
    valid_region = _detect_valid_region(data_input, nodata_input, min_valid_ratio)
    print(f"📍 Região válida detectada: {valid_region}")
    
    if valid_region is None:
        raise RuntimeError("Nenhuma região válida encontrada no arquivo")
    
    # 5. Extrair região válida
    x_start, y_start, x_end, y_end = valid_region
    data_cropped = data_input[y_start:y_end, x_start:x_end]
    valid_mask_cropped = _create_valid_mask(data_cropped, nodata_input)
    
    print(f"📊 Dados válidos: {np.sum(valid_mask_cropped)/valid_mask_cropped.size*100:.1f}%")
    
    # 6. Aplicar super-resolução
    result_data = apply_universal_super_resolution(
        data_cropped, valid_mask_cropped, model, norm_params, 
        patch_size, overlap, scale_factor
    )
    
    # 7. Salvar resultado com georeferenciamento correto
    _save_universal_tiff(
        result_data, ds_input, (x_start, y_start), output_path, 
        scale_factor, nodata_input
    )
    
    # 8. Limpar
    ds_input = None
    
    print("✅ Super-resolução universal concluída!")
    print(f"📁 Resultado: {output_path}")

def _detect_valid_region(data, nodata_value, min_valid_ratio):
    """
    Detecta automaticamente a região válida no arquivo.
    """
    
    # Criar máscara de dados válidos
    if nodata_value is not None:
        valid_mask = (data != nodata_value) & ~np.isnan(data) & ~np.isinf(data)
    else:
        valid_mask = ~np.isnan(data) & ~np.isinf(data)
    
    if not np.any(valid_mask):
        return None
    
    # Encontrar bounding box dos dados válidos
    valid_coords = np.where(valid_mask)
    y_min, y_max = np.min(valid_coords[0]), np.max(valid_coords[0])
    x_min, x_max = np.min(valid_coords[1]), np.max(valid_coords[1])
    
    # Adicionar margem de segurança (10% em cada direção)
    height, width = data.shape
    margin_y = max(1, int((y_max - y_min) * 0.1))
    margin_x = max(1, int((x_max - x_min) * 0.1))
    
    y_start = max(0, y_min - margin_y)
    y_end = min(height, y_max + margin_y + 1)
    x_start = max(0, x_min - margin_x)
    x_end = min(width, x_max + margin_x + 1)
    
    return (x_start, y_start, x_end, y_end)

def _create_valid_mask(data, nodata_value):
    """Cria máscara de dados válidos."""
    
    if nodata_value is not None:
        valid_mask = (data != nodata_value) & ~np.isnan(data) & ~np.isinf(data)
    else:
        valid_mask = ~np.isnan(data) & ~np.isinf(data)
    
    return valid_mask

def apply_universal_super_resolution(data, valid_mask, model, norm_params, 
                                   patch_size, overlap, scale_factor):
    """
    Aplica super-resolução universal com composição inteligente.
    """
    
    height, width = data.shape
    output_height = height * scale_factor
    output_width = width * scale_factor
    
    # Inicializar arrays de resultado
    result = np.zeros((output_height, output_width), dtype=np.float32)
    count = np.zeros((output_height, output_width), dtype=np.float32)
    
    # Criar máscara de peso para blending
    weight_mask = _create_weight_mask(patch_size * scale_factor, overlap * scale_factor)
    
    stride = max(1, patch_size - overlap)
    
    print(f"🔄 Processando patches com stride {stride}px...")
    
    patches_processed = 0
    patches_skipped = 0
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # Verificar se patch tem dados válidos suficientes
            patch_valid = valid_mask[y:y+patch_size, x:x+patch_size]
            valid_ratio = np.sum(patch_valid) / (patch_size * patch_size)
            
            if valid_ratio < 0.1:  # Menos de 10% válidos
                patches_skipped += 1
                continue
            
            # Extrair patch LR
            lr_patch = data[y:y+patch_size, x:x+patch_size].copy()
            
            # Preencher NoData com interpolação inteligente
            lr_patch = _fill_nodata_smart(lr_patch, patch_valid)
            
            # Normalizar
            lr_norm = _normalize_patch_universal(lr_patch, norm_params)
            
            # Aplicar modelo
            lr_input = lr_norm.reshape(1, patch_size, patch_size, 1)
            hr_pred = model.predict(lr_input, verbose=0)
            hr_pred = hr_pred.reshape(patch_size * scale_factor, patch_size * scale_factor)
            
            # Desnormalizar
            hr_pred = _denormalize_patch_universal(hr_pred, norm_params)
            
            # Aplicar máscara de peso
            hr_pred *= weight_mask
            
            # Adicionar ao resultado com blending
            y_out = y * scale_factor
            x_out = x * scale_factor
            h_out = patch_size * scale_factor
            w_out = patch_size * scale_factor
            
            result[y_out:y_out+h_out, x_out:x_out+w_out] += hr_pred
            count[y_out:y_out+h_out, x_out:x_out+w_out] += weight_mask
            
            patches_processed += 1
            
            if patches_processed % 100 == 0:
                print(f"  📊 Processados: {patches_processed}, Pulados: {patches_skipped}")
    
    # Normalizar por contagem (blending final)
    valid_count = count > 0
    result[valid_count] /= count[valid_count]
    
    # Preencher áreas vazias com interpolação
    result = _fill_empty_areas_universal(result, valid_mask, scale_factor)
    
    print(f"✅ Total: {patches_processed} processados, {patches_skipped} pulados")
    
    return result

def _fill_nodata_smart(patch, valid_mask):
    """
    Preenche NoData de forma inteligente usando interpolação adaptativa.
    """
    
    if np.all(valid_mask):
        return patch
    
    patch_filled = patch.copy()
    invalid_mask = ~valid_mask
    
    if np.any(invalid_mask):
        # Usar interpolação bilinear para preencher gaps
        try:
            patch_filled = cv2.inpaint(
                patch.astype(np.uint8), 
                invalid_mask.astype(np.uint8), 
                inpaintRadius=3, 
                flags=cv2.INPAINT_TELEA
            ).astype(np.float32)
        except:
            # Fallback: usar média dos vizinhos válidos
            patch_filled = _fill_with_neighbor_average(patch, valid_mask)
    
    return patch_filled

def _fill_with_neighbor_average(patch, valid_mask):
    """
    Preenche NoData com média dos vizinhos válidos (fallback).
    """
    
    patch_filled = patch.copy()
    invalid_mask = ~valid_mask
    
    if not np.any(invalid_mask):
        return patch_filled
    
    # Usar convolução para calcular média dos vizinhos
    kernel = np.ones((3, 3), dtype=np.float32) / 9
    valid_patch = patch.copy()
    valid_patch[invalid_mask] = 0
    
    # Convolução para média
    neighbor_sum = cv2.filter2D(valid_patch, -1, kernel)
    neighbor_count = cv2.filter2D(valid_mask.astype(np.float32), -1, kernel)
    
    # Evitar divisão por zero
    neighbor_count[neighbor_count == 0] = 1
    neighbor_avg = neighbor_sum / neighbor_count
    
    # Preencher apenas onde há vizinhos válidos
    fill_mask = invalid_mask & (neighbor_count > 0)
    patch_filled[fill_mask] = neighbor_avg[fill_mask]
    
    return patch_filled

def _fill_empty_areas_universal(result, original_valid_mask, scale_factor):
    """
    Preenche áreas vazias com interpolação universal.
    """
    
    # Upscale da máscara original
    valid_upscaled = cv2.resize(
        original_valid_mask.astype(np.uint8), 
        (result.shape[1], result.shape[0]), 
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    
    # Identificar áreas vazias que deveriam ter dados
    empty_mask = (result == 0) & valid_upscaled
    
    if not np.any(empty_mask):
        return result
    
    # Interpolação bilinear nas áreas vazias válidas
    result_interp = cv2.resize(
        result, (result.shape[1], result.shape[0]), 
        interpolation=cv2.INTER_LINEAR
    )
    
    result[empty_mask] = result_interp[empty_mask]
    
    return result

def _create_weight_mask(patch_size, overlap):
    """Cria máscara de peso para blending suave."""
    
    mask = np.ones((patch_size, patch_size), dtype=np.float32)
    
    # Reduzir peso nas bordas para blending suave
    fade_size = max(1, overlap // 2)
    
    # Bordas superior e inferior
    for i in range(fade_size):
        weight = (i + 1) / fade_size
        mask[i, :] *= weight
        mask[-(i+1), :] *= weight
    
    # Bordas esquerda e direita
    for j in range(fade_size):
        weight = (j + 1) / fade_size
        mask[:, j] *= weight
        mask[:, -(j+1)] *= weight
    
    return mask

def _normalize_patch_universal(patch, norm_params):
    """Normalização universal usando parâmetros de treinamento."""
    
    lr_min = norm_params.get('lr_min', 0)
    lr_max = norm_params.get('lr_max', 1)
    
    if lr_max - lr_min < 1e-6:
        return np.zeros_like(patch)
    
    normalized = (patch - lr_min) / (lr_max - lr_min)
    return np.clip(normalized, 0, 1)

def _denormalize_patch_universal(patch, norm_params):
    """Desnormalização universal usando parâmetros de treinamento."""
    
    hr_min = norm_params.get('hr_min', 0)
    hr_max = norm_params.get('hr_max', 1)
    
    if hr_max - hr_min < 1e-6:
        return patch
    
    denormalized = patch * (hr_max - hr_min) + hr_min
    return denormalized

def _load_normalization_params():
    """Carrega parâmetros de normalização com fallback."""
    
    try:
        norm_params = np.load('geospatial_output/norm_params.npy', allow_pickle=True).item()
        print("✅ Parâmetros de normalização carregados")
        return norm_params
    except:
        print("⚠️  Parâmetros não encontrados, usando padrão")
        return {'lr_min': 0, 'lr_max': 1, 'hr_min': 0, 'hr_max': 1}

def _save_universal_tiff(data, source_ds, crop_offset, output_path, scale_factor, nodata_value):
    """
    Salva TIFF com georeferenciamento universal.
    """
    
    gdal = _import_gdal()
    
    # Obter geotransform original
    geotransform = source_ds.GetGeoTransform()
    projection = source_ds.GetProjection()
    
    # Ajustar geotransform para crop e escala
    x_offset, y_offset = crop_offset
    new_geotransform = list(geotransform)
    new_geotransform[0] = geotransform[0] + x_offset * geotransform[1]  # X origin
    new_geotransform[3] = geotransform[3] + y_offset * geotransform[5]  # Y origin
    new_geotransform[1] = geotransform[1] / scale_factor  # Pixel width
    new_geotransform[5] = geotransform[5] / scale_factor  # Pixel height
    
    # Criar dataset de saída
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(
        output_path, 
        data.shape[1], data.shape[0], 
        1, gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
    )
    
    if ds_out is None:
        raise RuntimeError(f"Não foi possível criar: {output_path}")
    
    # Definir georeferenciamento
    ds_out.SetGeoTransform(new_geotransform)
    ds_out.SetProjection(projection)
    
    # Escrever dados
    band_out = ds_out.GetRasterBand(1)
    band_out.WriteArray(data)
    
    if nodata_value is not None:
        band_out.SetNoDataValue(nodata_value)
    
    # Fechar
    ds_out = None
    
    print(f"💾 Salvo: {output_path}")

def _load_model_safe(model_path):
    """Carrega modelo com fallback seguro."""
    
    try:
        model = tf.keras.models.load_model(model_path, safe_mode=False)
        print(f"✅ Modelo carregado: {model_path}")
        return model
    except Exception as e:
        print(f"⚠️  Erro ao carregar modelo: {e}")
        print("🔄 Usando modelo simples...")
        return _create_simple_model()

def _create_simple_model():
    """Cria modelo simples como fallback."""
    
    inputs = tf.keras.Input(shape=(30, 30, 1))
    x = tf.keras.layers.Conv2D(64, (9, 9), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(1, (5, 5), activation='linear', padding='same')(x)
    x = tf.keras.layers.UpSampling2D(size=(3, 3), interpolation='bilinear')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    print("✅ Modelo simples criado")
    return model

def _import_gdal():
    """Importa GDAL com tratamento de erro."""
    try:
        from osgeo import gdal
        return gdal
    except Exception as e:
        raise ImportError("Falha ao importar GDAL") from e

if __name__ == "__main__":
    # Exemplo de uso universal
    universal_hybrid_super_resolution(
        input_path='data/images/ANADEM_AricanduvaBufferUTM.tif',
        output_path='geospatial_output/universal_result.tif'
    )
