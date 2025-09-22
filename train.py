#!/usr/bin/env python3
"""
Treinamento melhorado com dados reais e técnicas avançadas.
"""

import numpy as np
import tensorflow as tf
from osgeo import gdal
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def create_real_lr_hr_pairs(anadem_path, geosampa_path, output_dir, patch_size=15):
    """
    Cria pares LR-HR REAIS usando ANADEM (LR) e GEOSAMPA (HR).
    Abordagem simplificada: usar GEOSAMPA para criar LR artificial + HR real.
    """
    
    print("🎯 CRIANDO DADOS REAIS LR-HR")
    print("=" * 50)
    
    # Carregar GEOSAMPA (usar como fonte de HR real)
    ds_geosampa = gdal.Open(geosampa_path)
    if ds_geosampa is None:
        raise RuntimeError("Não foi possível abrir GEOSAMPA")
    
    # Usar região válida conhecida do GEOSAMPA (otimizada para 40x)
    x_start, y_start = 18437, 13802
    region_width, region_height = 800, 800  # Suficiente para patches 600x600
    
    print(f"📍 Usando região GEOSAMPA: ({x_start}, {y_start}) - {region_width}x{region_height}")
    
    # Extrair região do GEOSAMPA
    geosampa_data = ds_geosampa.GetRasterBand(1).ReadAsArray(
        x_start, y_start, region_width, region_height)
    
    if geosampa_data is None:
        raise RuntimeError("Não foi possível ler região do GEOSAMPA")
    
    # Criar máscara de dados válidos
    nodata_value = ds_geosampa.GetRasterBand(1).GetNoDataValue()
    if nodata_value is not None:
        valid_mask = (geosampa_data != nodata_value) & ~np.isnan(geosampa_data) & ~np.isinf(geosampa_data)
    else:
        valid_mask = ~np.isnan(geosampa_data) & ~np.isinf(geosampa_data)
    
    print(f"📊 Dados válidos: {np.sum(valid_mask)/valid_mask.size*100:.1f}%")
    
    # Extrair patches HR do GEOSAMPA
    patches_lr = []
    patches_hr = []
    
    stride = patch_size * 2  # Overlap balanceado para 3x
    scale_factor = 3  # 3x super-resolution (30m → 10m)
    
    print(f"🔍 Extraindo patches: stride={stride}, patch_size={patch_size}, scale_factor={scale_factor}")
    print(f"🔍 Range Y: 0 a {region_height - patch_size * scale_factor + 1}")
    print(f"🔍 Range X: 0 a {region_width - patch_size * scale_factor + 1}")
    
    patch_count = 0
    for y in range(0, region_height - patch_size * scale_factor + 1, stride):
        for x in range(0, region_width - patch_size * scale_factor + 1, stride):
            patch_count += 1
            if patch_count % 10 == 0:
                print(f"🔍 Processando patch {patch_count} em ({x}, {y})")
            
            # Extrair patch HR (45x45 do GEOSAMPA)
            hr_patch = geosampa_data[y:y+patch_size*scale_factor, x:x+patch_size*scale_factor]
            
            # Verificar se patch tem dados válidos
            patch_valid = valid_mask[y:y+patch_size*scale_factor, x:x+patch_size*scale_factor]
            valid_ratio = np.sum(patch_valid) / (patch_size * scale_factor)**2
            if valid_ratio < 0.8:
                if patch_count <= 5:  # Debug apenas os primeiros
                    print(f"❌ Patch {patch_count} rejeitado: {valid_ratio:.2%} válido")
                continue
            
            # Criar LR artificial (15x15) a partir do HR
            lr_patch = create_realistic_lr_from_hr(hr_patch, scale_factor)
            
            # Validar par
            if validate_real_patch_pair(lr_patch, hr_patch):
                patches_lr.append(lr_patch)
                patches_hr.append(hr_patch)
                if len(patches_lr) <= 5:  # Debug apenas os primeiros
                    print(f"✅ Patch {patch_count} aceito: {valid_ratio:.2%} válido")
            else:
                if patch_count <= 5:  # Debug apenas os primeiros
                    print(f"❌ Patch {patch_count} falhou na validação")
    
    print(f"🔍 Total de posições testadas: {patch_count}")
    
    print(f"✅ Criados {len(patches_lr)} pares LR-HR reais")
    
    if len(patches_lr) == 0:
        print("⚠️  Nenhum par válido encontrado. Usando fallback...")
        return create_fallback_pairs(geosampa_data, valid_mask, output_dir, patch_size, scale_factor)
    
    # Salvar patches
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (lr, hr) in enumerate(zip(patches_lr, patches_hr)):
        np.save(os.path.join(output_dir, f"real_lr_{i:06d}.npy"), lr)
        np.save(os.path.join(output_dir, f"real_hr_{i:06d}.npy"), hr)
    
    ds_geosampa = None
    
    return len(patches_lr)

def create_realistic_lr_from_hr(hr_patch, scale_factor):
    """
    Cria LR realístico a partir do HR usando downsampling inteligente.
    """
    
    # Método 1: Downsampling com blur (mais realístico)
    # Aplicar blur gaussiano antes do downsampling (kernel deve ser ímpar)
    kernel_size = min(scale_factor, 15)  # Limitar tamanho do kernel
    if kernel_size % 2 == 0:
        kernel_size += 1  # Garantir que seja ímpar
    
    blurred = cv2.GaussianBlur(hr_patch, (kernel_size, kernel_size), 0)
    
    # Downsample para 15x15 (3x super-resolution)
    lr_small = cv2.resize(blurred, (15, 15), interpolation=cv2.INTER_AREA)
    
    # Método 2: Adicionar ruído realístico
    noise_std = np.std(lr_small) * 0.02  # 2% do desvio padrão
    noise = np.random.normal(0, noise_std, lr_small.shape)
    lr_patch = lr_small + noise
    
    return lr_patch

def create_fallback_pairs(geosampa_data, valid_mask, output_dir, patch_size, scale_factor):
    """
    Fallback: criar pares usando apenas GEOSAMPA com downsampling.
    """
    
    print("🔄 Criando pares de fallback...")
    
    patches_lr = []
    patches_hr = []
    
    stride = patch_size * 2  # Usar mesmo stride do principal
    
    for y in range(0, geosampa_data.shape[0] - patch_size * scale_factor + 1, stride):
        for x in range(0, geosampa_data.shape[1] - patch_size * scale_factor + 1, stride):
            # Extrair patch HR
            hr_patch = geosampa_data[y:y+patch_size*scale_factor, x:x+patch_size*scale_factor]
            
            # Verificar validade
            patch_valid = valid_mask[y:y+patch_size*scale_factor, x:x+patch_size*scale_factor]
            if np.sum(patch_valid) < (patch_size * scale_factor)**2 * 0.5:
                continue
            
            # Criar LR
            lr_patch = create_realistic_lr_from_hr(hr_patch, scale_factor)
            
            patches_lr.append(lr_patch)
            patches_hr.append(hr_patch)
            
            if len(patches_lr) >= 100:  # Reduzido para teste
                break
        if len(patches_lr) >= 100:
            break
    
    print(f"✅ Fallback: {len(patches_lr)} pares criados")
    
    # Salvar patches
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (lr, hr) in enumerate(zip(patches_lr, patches_hr)):
        np.save(os.path.join(output_dir, f"real_lr_{i:06d}.npy"), lr)
        np.save(os.path.join(output_dir, f"real_hr_{i:06d}.npy"), hr)
    
    return len(patches_lr)

def find_overlap_region(ds_anadem, ds_geosampa):
    """Encontra região de sobreposição geográfica entre ANADEM e GEOSAMPA."""
    
    # Obter geotransforms
    gt_anadem = ds_anadem.GetGeoTransform()
    gt_geosampa = ds_geosampa.GetGeoTransform()
    
    # Calcular coordenadas geográficas dos cantos
    anadem_bounds = get_geographic_bounds(ds_anadem, gt_anadem)
    geosampa_bounds = get_geographic_bounds(ds_geosampa, gt_geosampa)
    
    # Encontrar interseção
    x_min = max(anadem_bounds[0], geosampa_bounds[0])
    y_min = max(anadem_bounds[1], geosampa_bounds[1])
    x_max = min(anadem_bounds[2], geosampa_bounds[2])
    y_max = min(anadem_bounds[3], geosampa_bounds[3])
    
    if x_min >= x_max or y_min >= y_max:
        return None
    
    # Converter para coordenadas de pixel do ANADEM
    anadem_x = int((x_min - gt_anadem[0]) / gt_anadem[1])
    anadem_y = int((y_min - gt_anadem[3]) / gt_anadem[5])
    anadem_w = int((x_max - x_min) / gt_anadem[1])
    anadem_h = int((y_max - y_min) / abs(gt_anadem[5]))
    
    return (anadem_x, anadem_y, anadem_w, anadem_h)

def get_geographic_bounds(ds, geotransform):
    """Calcula limites geográficos de um dataset."""
    width = ds.RasterXSize
    height = ds.RasterYSize
    
    x_min = geotransform[0]
    y_max = geotransform[3]
    x_max = geotransform[0] + width * geotransform[1]
    y_min = geotransform[3] + height * geotransform[5]
    
    return (x_min, y_min, x_max, y_max)

def validate_real_patch_pair(lr_patch, hr_patch):
    """Valida par de patches LR-HR reais."""
    
    # Verificar dimensões (15x15 LR, 45x45 HR para 3x)
    if lr_patch.shape != (15, 15) or hr_patch.shape != (45, 45):
        return False
    
    # Verificar dados válidos (mais permissivo)
    lr_valid = ~np.isnan(lr_patch) & ~np.isinf(lr_patch)
    hr_valid = ~np.isnan(hr_patch) & ~np.isinf(hr_patch)
    
    if np.sum(lr_valid) < lr_patch.size * 0.5 or np.sum(hr_valid) < hr_patch.size * 0.5:
        return False
    
    # Verificar se há variação nos dados (não uniforme)
    lr_std = np.std(lr_patch[lr_valid]) if np.sum(lr_valid) > 0 else 0
    hr_std = np.std(hr_patch[hr_valid]) if np.sum(hr_valid) > 0 else 0
    
    if lr_std < 1e-6 or hr_std < 1e-6:  # Dados muito uniformes
        return False
    
    return True

def create_advanced_srcnn_model(input_shape=(15, 15, 1), scale_factor=3):
    """
    Cria modelo SRCNN avançado com técnicas modernas para 3x super-resolution.
    """
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Feature extraction com múltiplas escalas
    x1 = tf.keras.layers.Conv2D(64, (9, 9), activation='relu', padding='same')(inputs)
    x2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
    x3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    
    # Concatenar features multi-escala
    x = tf.keras.layers.Concatenate()([x1, x2, x3])
    
    # Non-linear mapping com skip connections
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Upscaling usando UpSampling2D (compatível com serialização)
    x = tf.keras.layers.UpSampling2D(size=(scale_factor, scale_factor), interpolation='bilinear')(x)
    
    # Refinamento final
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    output = tf.keras.layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    
    return model

def train_advanced_model(patches_dir, epochs=10, batch_size=32):
    """
    Treina modelo avançado com técnicas modernas.
    """
    
    print("🧠 TREINAMENTO AVANÇADO")
    print("=" * 40)
    
    # Carregar dados reais
    lr_patches, hr_patches = load_real_patches(patches_dir)
    
    # Normalização robusta
    lr_norm, hr_norm, norm_params = robust_normalization(lr_patches, hr_patches)
    
    # Split com estratificação por variabilidade
    lr_train, lr_val, hr_train, hr_val = stratified_split(lr_norm, hr_norm)
    
    # Criar modelo avançado
    model = create_advanced_srcnn_model()
    
    # Compilar com otimizações
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Callbacks para treinamento completo
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'geospatial_output/advanced_model_best.keras',
            monitor='val_loss', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )
    ]
    
    # Treinar
    history = model.fit(
        lr_train, hr_train,
        validation_data=(lr_val, hr_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Salvar parâmetros de normalização
    np.save('geospatial_output/norm_params.npy', norm_params)
    
    return model, history, norm_params

def load_real_patches(patches_dir):
    """Carrega patches reais LR-HR com shape correto."""
    lr_files = sorted([f for f in os.listdir(patches_dir) if f.startswith('real_lr_')])
    hr_files = sorted([f for f in os.listdir(patches_dir) if f.startswith('real_hr_')])
    lr_patches = []
    hr_patches = []
    for lr_file, hr_file in zip(lr_files, hr_files):
        lr_patch = np.load(os.path.join(patches_dir, lr_file))
        hr_patch = np.load(os.path.join(patches_dir, hr_file))
        if lr_patch.shape == (15, 15) and hr_patch.shape == (45, 45):
            lr_patches.append(lr_patch)
            hr_patches.append(hr_patch)
    if len(lr_patches) == 0 or len(hr_patches) == 0:
        raise ValueError("Nenhum patch válido encontrado (shape incorreto)")
    return np.array(lr_patches), np.array(hr_patches)

def robust_normalization(lr_patches, hr_patches):
    """Normalização robusta com múltiplas estratégias."""
    
    # Verificar se há patches
    if len(lr_patches) == 0 or len(hr_patches) == 0:
        raise ValueError("Nenhum patch encontrado para normalização")
    
    # Calcular estatísticas globais
    all_lr = np.concatenate([p.flatten() for p in lr_patches])
    all_hr = np.concatenate([p.flatten() for p in hr_patches])
    
    # Filtrar valores inválidos
    all_lr = all_lr[~np.isnan(all_lr) & ~np.isinf(all_lr)]
    all_hr = all_hr[~np.isnan(all_hr) & ~np.isinf(all_hr)]
    
    if len(all_lr) == 0 or len(all_hr) == 0:
        raise ValueError("Nenhum dado válido encontrado nos patches")
    
    # Usar percentis para robustez
    lr_min, lr_max = np.percentile(all_lr, [1, 99])
    hr_min, hr_max = np.percentile(all_hr, [1, 99])
    
    # Verificar se range é válido
    if lr_max - lr_min < 1e-6:
        lr_min, lr_max = np.min(all_lr), np.max(all_lr)
    if hr_max - hr_min < 1e-6:
        hr_min, hr_max = np.min(all_hr), np.max(all_hr)
    
    # Normalizar
    lr_norm = (lr_patches - lr_min) / (lr_max - lr_min)
    hr_norm = (hr_patches - hr_min) / (hr_max - hr_min)
    
    # Clipping para evitar outliers
    lr_norm = np.clip(lr_norm, 0, 1)
    hr_norm = np.clip(hr_norm, 0, 1)
    
    norm_params = {
        'lr_min': lr_min, 'lr_max': lr_max,
        'hr_min': hr_min, 'hr_max': hr_max
    }
    
    print(f"📊 Normalização: LR [{lr_min:.2f}, {lr_max:.2f}], HR [{hr_min:.2f}, {hr_max:.2f}]")
    
    return lr_norm, hr_norm, norm_params

def stratified_split(lr_patches, hr_patches, test_size=0.2):
    """Split estratificado baseado na variabilidade dos patches."""
    
    # Calcular variabilidade de cada patch
    variability = []
    for lr, hr in zip(lr_patches, hr_patches):
        lr_var = np.var(lr)
        hr_var = np.var(hr)
        variability.append(lr_var + hr_var)
    
    variability = np.array(variability)
    
    # Dividir em quartis para estratificação
    quartiles = np.percentile(variability, [25, 50, 75])
    
    strata = np.zeros_like(variability)
    strata[variability <= quartiles[0]] = 0  # Baixa variabilidade
    strata[(variability > quartiles[0]) & (variability <= quartiles[1])] = 1
    strata[(variability > quartiles[1]) & (variability <= quartiles[2])] = 2
    strata[variability > quartiles[2]] = 3  # Alta variabilidade
    
    # Split estratificado
    lr_train, lr_val, hr_train, hr_val = train_test_split(
        lr_patches, hr_patches, test_size=test_size, 
        stratify=strata, random_state=42
    )
    
    return lr_train, lr_val, hr_train, hr_val

def main():
    """Pipeline de treinamento melhorado."""
    
    print("🚀 PIPELINE DE TREINAMENTO MELHORADO (3X SUPER-RESOLUTION)")
    print("=" * 60)
    print("⚡ Modo teste: 5 épocas, patches 15x15→45x45, região 800x800")
    
    # 1. Criar dados reais LR-HR
    print("1️⃣ Criando dados reais...")
    num_patches = create_real_lr_hr_pairs(
        anadem_path='data/images/ANADEM_AricanduvaBufferUTM.tif',
        geosampa_path='data/images/MDTGeosampa_AricanduvaBufferUTM.tif',
        output_dir='geospatial_output/real_patches'
    )
    
    # 2. Treinar modelo avançado
    print("2️⃣ Treinando modelo avançado...")
    model, history, norm_params = train_advanced_model(
        patches_dir='geospatial_output/real_patches',
        epochs=15,
        batch_size=16
    )
    
    print("✅ Treinamento melhorado concluído!")
    print(f"📊 Patches reais: {num_patches}")
    print(f"📁 Modelo: geospatial_output/advanced_model_best.keras")

if __name__ == "__main__":
    main()