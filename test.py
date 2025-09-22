#!/usr/bin/env python3
"""
Super-resolução híbrida: Modelo SRCNN treinado + Interpolação bilinear
Combina o conhecimento do modelo com a suavidade da interpolação.
"""

import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import cv2
import os

def _import_gdal():
    """Importa GDAL com tratamento de erro."""
    try:
        from osgeo import gdal
        return gdal
    except Exception as e:
        raise ImportError("Falha ao importar GDAL (osgeo)") from e

def _load_srcnn_model(model_path: str):
    """Carrega apenas o modelo SRCNN treinado (sem upscaling)."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modelo SRCNN carregado: {model_path}")
        print(f"   Input: {model.input_shape}")
        print(f"   Output: {model.output_shape}")
        return model
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo: {e}")

def _load_normalization_params():
    """Carregar parâmetros de normalização salvos do treinamento."""
    norm_path = 'geospatial_output/norm_params.npy'
    
    if not os.path.exists(norm_path):
        raise RuntimeError(f"Parâmetros de normalização não encontrados: {norm_path}")
    
    norm_params = np.load(norm_path, allow_pickle=True).item()
    
    # Usar parâmetros do HR (que é o que o modelo gera)
    data_min = norm_params['hr_min']
    data_max = norm_params['hr_max']
    
    print(f"📊 Parâmetros de normalização carregados:")
    print(f"   HR Min: {data_min:.2f}")
    print(f"   HR Max: {data_max:.2f}")
    
    return data_min, data_max

def _normalize_patch(patch: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
    """Normaliza patch usando parâmetros do GEOSAMPA."""
    eps = 1e-12
    range_val = max(eps, (data_max - data_min))
    normalized = (patch - data_min) / range_val
    return np.clip(normalized, 0.0, 1.0)

def _denormalize_patch(patch: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
    """Desnormaliza patch para escala original."""
    return patch * (data_max - data_min) + data_min

def hybrid_anadem_super_resolution(
    anadem_path='data/images/ANADEM_AricanduvaBufferUTM.tif',
    model_path='geospatial_output/advanced_model_best.keras',
    output_path='geospatial_output/anadem_hybrid_super_resolution_3x.tif',
    scale_factor=3,
    patch_size=15,
    overlap=7,
    use_model_weight=0.7  # 70% modelo, 30% interpolação
):
    """
    Aplica super-resolução híbrida: modelo + interpolação (3x).
    
    Args:
        anadem_path: Caminho do ANADEM
        model_path: Caminho do modelo treinado (3x)
        output_path: Caminho de saída
        scale_factor: Fator de upscaling (3x)
        patch_size: Tamanho dos patches (15x15)
        overlap: Sobreposição entre patches
        use_model_weight: Peso do modelo vs interpolação (0.0-1.0)
    """
    
    print("🚀 SUPER-RESOLUÇÃO HÍBRIDA - ANADEM")
    print("=" * 50)
    print(f"📁 ANADEM: {anadem_path}")
    print(f"🧠 Modelo: {model_path}")
    print(f"📁 Saída: {output_path}")
    print(f"🔧 Método: {use_model_weight*100:.0f}% Modelo + {(1-use_model_weight)*100:.0f}% Interpolação")
    print(f"🔍 Scale factor: {scale_factor}x")
    
    gdal = _import_gdal()
    
    # 1. Carregar modelo SRCNN
    srcnn_model = _load_srcnn_model(model_path)
    
    # 2. Obter parâmetros de normalização
    data_min, data_max = _load_normalization_params()
    
    # 3. Carregar ANADEM
    ds_anadem = gdal.Open(anadem_path)
    if ds_anadem is None:
        raise RuntimeError(f"Não foi possível abrir: {anadem_path}")
    
    band_anadem = ds_anadem.GetRasterBand(1)
    data_anadem = band_anadem.ReadAsArray()
    nodata_anadem = band_anadem.GetNoDataValue()
    geotransform = ds_anadem.GetGeoTransform()
    projection = ds_anadem.GetProjection()
    
    height, width = data_anadem.shape
    print(f"📊 ANADEM: {width} x {height} pixels")
    
    # 4. Verificar compatibilidade
    valid_mask = ~np.isnan(data_anadem) & ~np.isinf(data_anadem)
    if nodata_anadem is not None:
        valid_mask = valid_mask & (data_anadem != nodata_anadem)
    
    valid_data = data_anadem[valid_mask]
    if len(valid_data) > 0:
        anadem_min, anadem_max = np.min(valid_data), np.max(valid_data)
        print(f"📊 ANADEM range: {anadem_min:.2f} - {anadem_max:.2f}")
        
        if anadem_min < data_min or anadem_max > data_max:
            print("⚠️  ANADEM fora do range de treinamento - resultados podem variar")
    
    # 5. MÉTODO 1: Super-resolução usando modelo SRCNN (patch por patch)
    print("🧠 Aplicando modelo SRCNN...")
    
    # Usar stride menor para melhor cobertura
    stride = max(1, patch_size - overlap)
    num_patches_x = (width - patch_size) // stride + 1
    num_patches_y = (height - patch_size) // stride + 1
    
    print(f"📊 Processando {num_patches_x} x {num_patches_y} = {num_patches_x * num_patches_y} patches")
    
    model_result = np.zeros((height, width), dtype=np.float32)
    model_weights = np.zeros((height, width), dtype=np.float32)
    
    processed_patches = 0
    valid_patches = 0
    
    for j in range(num_patches_y):
        for i in range(num_patches_x):
            x = i * stride
            y = j * stride
            
            # Verificar limites
            if x + patch_size > width or y + patch_size > height:
                continue
            
            try:
                # Extrair patch
                patch = data_anadem[y:y+patch_size, x:x+patch_size]
                
                # Verificar dados válidos (critério mais relaxado)
                patch_valid_mask = ~np.isnan(patch) & ~np.isinf(patch)
                if nodata_anadem is not None:
                    patch_valid_mask = patch_valid_mask & (patch != nodata_anadem)
                
                # Relaxar critério: aceitar patches com pelo menos 20% de dados válidos
                if np.sum(patch_valid_mask) < patch_size * patch_size * 0.2:
                    continue
                
                # Interpolar valores inválidos de forma mais robusta
                if np.sum(~patch_valid_mask) > 0:
                    valid_values = patch[patch_valid_mask]
                    if len(valid_values) > 0:
                        # Usar interpolação espacial em vez de média simples
                        from scipy import ndimage
                        try:
                            # Interpolação por distância inversa
                            patch_interp = patch.copy()
                            patch_interp[~patch_valid_mask] = 0
                            patch_interp = ndimage.gaussian_filter(patch_interp, sigma=1.0)
                            patch[~patch_valid_mask] = patch_interp[~patch_valid_mask]
                        except:
                            # Fallback para média se scipy não estiver disponível
                            patch[~patch_valid_mask] = np.mean(valid_values)
                
                # Normalizar
                patch_norm = _normalize_patch(patch, data_min, data_max)
                
                # Aplicar modelo
                patch_input = patch_norm[..., None][None, ...]
                enhanced_patch = srcnn_model.predict(patch_input, verbose=0)[0, :, :, 0]
                
                # Desnormalizar
                enhanced_patch = _denormalize_patch(enhanced_patch, data_min, data_max)
                
                # Adicionar ao resultado com pesos
                weight_mask = np.ones((patch_size, patch_size))
                if overlap > 0:
                    for margin in range(min(overlap, patch_size//4)):
                        weight_mask[margin, :] *= 0.5
                        weight_mask[-margin-1, :] *= 0.5
                        weight_mask[:, margin] *= 0.5
                        weight_mask[:, -margin-1] *= 0.5
                
                model_result[y:y+patch_size, x:x+patch_size] += enhanced_patch * weight_mask
                model_weights[y:y+patch_size, x:x+patch_size] += weight_mask
                
                valid_patches += 1
                
            except Exception as e:
                continue
            
            processed_patches += 1
            if processed_patches % 50 == 0:
                print(f"   Processados {processed_patches} patches...")
    
    # Normalizar resultado do modelo
    valid_model_mask = model_weights > 0
    model_result[valid_model_mask] = model_result[valid_model_mask] / model_weights[valid_model_mask]
    
    print(f"✅ Modelo aplicado: {valid_patches}/{processed_patches} patches válidos")
    
    # Processar patches adicionais nas bordas para melhor cobertura
    print("🔄 Processando patches de borda...")
    edge_patches = 0
    
    # Patches nas bordas direita e inferior
    for j in range(num_patches_y):
        # Borda direita
        x = width - patch_size
        y = j * stride
        if y + patch_size <= height:
            try:
                patch = data_anadem[y:y+patch_size, x:x+patch_size]
                patch_valid_mask = ~np.isnan(patch) & ~np.isinf(patch)
                if nodata_anadem is not None:
                    patch_valid_mask = patch_valid_mask & (patch != nodata_anadem)
                
                if np.sum(patch_valid_mask) >= patch_size * patch_size * 0.2:
                    # Interpolar valores inválidos
                    if np.sum(~patch_valid_mask) > 0:
                        valid_values = patch[patch_valid_mask]
                        if len(valid_values) > 0:
                            patch[~patch_valid_mask] = np.mean(valid_values)
                    
                    # Aplicar modelo
                    patch_norm = _normalize_patch(patch, data_min, data_max)
                    patch_input = patch_norm[..., None][None, ...]
                    enhanced_patch = srcnn_model.predict(patch_input, verbose=0)[0, :, :, 0]
                    enhanced_patch = _denormalize_patch(enhanced_patch, data_min, data_max)
                    
                    # Adicionar com peso reduzido (borda)
                    weight_mask = np.ones((patch_size, patch_size)) * 0.5
                    model_result[y:y+patch_size, x:x+patch_size] += enhanced_patch * weight_mask
                    model_weights[y:y+patch_size, x:x+patch_size] += weight_mask
                    edge_patches += 1
            except:
                pass
    
    # Borda inferior
    for i in range(num_patches_x):
        x = i * stride
        y = height - patch_size
        if x + patch_size <= width:
            try:
                patch = data_anadem[y:y+patch_size, x:x+patch_size]
                patch_valid_mask = ~np.isnan(patch) & ~np.isinf(patch)
                if nodata_anadem is not None:
                    patch_valid_mask = patch_valid_mask & (patch != nodata_anadem)
                
                if np.sum(patch_valid_mask) >= patch_size * patch_size * 0.2:
                    # Interpolar valores inválidos
                    if np.sum(~patch_valid_mask) > 0:
                        valid_values = patch[patch_valid_mask]
                        if len(valid_values) > 0:
                            patch[~patch_valid_mask] = np.mean(valid_values)
                    
                    # Aplicar modelo
                    patch_norm = _normalize_patch(patch, data_min, data_max)
                    patch_input = patch_norm[..., None][None, ...]
                    enhanced_patch = srcnn_model.predict(patch_input, verbose=0)[0, :, :, 0]
                    enhanced_patch = _denormalize_patch(enhanced_patch, data_min, data_max)
                    
                    # Adicionar com peso reduzido (borda)
                    weight_mask = np.ones((patch_size, patch_size)) * 0.5
                    model_result[y:y+patch_size, x:x+patch_size] += enhanced_patch * weight_mask
                    model_weights[y:y+patch_size, x:x+patch_size] += weight_mask
                    edge_patches += 1
            except:
                pass
    
    print(f"✅ Patches de borda processados: {edge_patches}")
    
    # 6. MÉTODO 2: Interpolação bilinear simples
    print("📐 Aplicando interpolação bilinear...")
    
    # Preparar dados para interpolação
    data_for_interp = data_anadem.copy().astype(np.float64)
    if nodata_anadem is not None:
        data_for_interp[data_anadem == nodata_anadem] = np.nan
    
    # Fazer upscaling bilinear
    new_height = height * scale_factor
    new_width = width * scale_factor
    interp_result = cv2.resize(data_for_interp, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # 7. MÉTODO 3: Combinar modelo + interpolação com cobertura melhorada
    print("🔄 Combinando resultados...")
    
    # Fazer upscaling do resultado do modelo
    model_upscaled = cv2.resize(model_result, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Fazer upscaling dos pesos do modelo
    model_weights_upscaled = cv2.resize(model_weights, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Combinar usando pesos suavizados
    final_result = np.zeros_like(interp_result)
    
    # Máscara de dados válidos na interpolação
    valid_interp_mask = ~np.isnan(interp_result) & ~np.isinf(interp_result)
    
    # Máscara de cobertura do modelo (suavizada)
    model_coverage_mask = model_weights_upscaled > 0.1  # Threshold mais baixo
    
    # Aplicar filtro gaussiano para suavizar transições
    try:
        from scipy import ndimage
        model_coverage_smooth = ndimage.gaussian_filter(model_coverage_mask.astype(np.float32), sigma=2.0)
    except:
        model_coverage_smooth = model_coverage_mask.astype(np.float32)
    
    # Normalizar pesos para 0-1
    if np.max(model_coverage_smooth) > 0:
        model_coverage_smooth = model_coverage_smooth / np.max(model_coverage_smooth)
    
    # Combinar usando pesos adaptativos
    final_result = (
        model_coverage_smooth * use_model_weight * model_upscaled + 
        (1 - model_coverage_smooth * use_model_weight) * interp_result
    )
    
    # Garantir que áreas sem dados válidos mantenham NoData
    final_result[~valid_interp_mask] = interp_result[~valid_interp_mask]
    
    # Restaurar NoData
    if nodata_anadem is not None:
        final_result[~valid_interp_mask] = nodata_anadem
    
    # 8. Verificar resultado final
    print("🔍 Verificando resultado...")
    final_valid_mask = ~np.isnan(final_result) & ~np.isinf(final_result)
    if nodata_anadem is not None:
        final_valid_mask = final_valid_mask & (final_result != nodata_anadem)
    
    final_valid_data = final_result[final_valid_mask]
    if len(final_valid_data) > 0:
        print(f"📊 Resultado final: {np.min(final_valid_data):.2f} - {np.max(final_valid_data):.2f}")
        print(f"📊 Cobertura modelo: {np.sum(model_coverage_mask)/model_coverage_mask.size*100:.1f}%")
        print(f"📊 Cobertura suavizada: {np.sum(model_coverage_smooth > 0.1)/model_coverage_smooth.size*100:.1f}%")
    
    # 9. Salvar resultado
    print("💾 Salvando resultado...")
    
    # Ajustar geotransform
    new_geotransform = list(geotransform)
    new_geotransform[1] = geotransform[1] / scale_factor
    new_geotransform[5] = geotransform[5] / scale_factor
    
    # Criar dataset de saída
    driver = gdal.GetDriverByName('GTiff')
    options = ['COMPRESS=LZW', 'TILED=YES']
    out_ds = driver.Create(output_path, new_width, new_height, 1, gdal.GDT_Float32, options=options)
    
    out_ds.SetGeoTransform(new_geotransform)
    out_ds.SetProjection(projection)
    
    out_band = out_ds.GetRasterBand(1)
    
    # LIMPEZA FINAL: Remover NaN/Inf antes de salvar
    print("🧹 Limpando dados finais (removendo NaN/Inf)...")
    final_result_clean = final_result.copy()
    
    # Identificar valores inválidos
    invalid_mask = np.isnan(final_result_clean) | np.isinf(final_result_clean)
    
    if np.sum(invalid_mask) > 0:
        print(f"⚠️  Encontrados {np.sum(invalid_mask)} valores inválidos")
        
        # Substituir por interpolação bilinear limpa
        if nodata_anadem is not None:
            final_result_clean[invalid_mask] = nodata_anadem
        else:
            # Usar valor médio dos dados válidos
            valid_data = final_result_clean[~invalid_mask]
            if len(valid_data) > 0:
                mean_value = np.mean(valid_data)
                final_result_clean[invalid_mask] = mean_value
            else:
                final_result_clean[invalid_mask] = 0.0
    
    # Verificar resultado final limpo
    final_clean_valid = ~np.isnan(final_result_clean) & ~np.isinf(final_result_clean)
    if np.sum(final_clean_valid) > 0:
        clean_data = final_result_clean[final_clean_valid]
        print(f"📊 Dados limpos: {np.min(clean_data):.2f} - {np.max(clean_data):.2f}")
    
    out_band.WriteArray(final_result_clean.astype(np.float32))
    if nodata_anadem is not None:
        out_band.SetNoDataValue(nodata_anadem)
    
    out_band.ComputeStatistics(False)
    out_ds.SetDescription(f'ANADEM Hybrid Super-Resolution ({scale_factor}x) - SRCNN + Bilinear')
    
    out_ds = None
    ds_anadem = None
    
    print(f"✅ Super-resolução híbrida concluída!")
    print(f"📁 Arquivo: {output_path}")
    print(f"📊 Resolução: {geotransform[1]:.1f}m → {new_geotransform[1]:.1f}m por pixel")
    
    return output_path

def main():
    """Função principal."""
    
    print("🎯 SUPER-RESOLUÇÃO HÍBRIDA - ANADEM")
    print("Combinando modelo SRCNN + interpolação bilinear")
    print("=" * 60)
    
    try:
        result_path = hybrid_anadem_super_resolution(
            anadem_path='data/images/ANADEM_AricanduvaBufferUTM.tif',
            model_path='geospatial_output/advanced_model_best.keras',
            output_path='geospatial_output/anadem_hybrid_super_resolution_3x.tif',
            scale_factor=3,
            use_model_weight=0.7  # 70% modelo, 30% interpolação
        )
        
        print(f"\n🎉 SUCESSO!")
        print(f"📁 Resultado: {result_path}")
        print(f"\n💡 O resultado combina:")
        print(f"   • 70% Conhecimento do modelo SRCNN")
        print(f"   • 30% Suavidade da interpolação bilinear")
        print(f"\n✅ Abra no QGIS para verificar a qualidade!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()
