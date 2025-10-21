#!/usr/bin/env python3
"""
Treinamento (variante RAW CROP) com filtro estrito:
- Primeiro recorta patches no GEOSAMPA em resolução nativa (ex.: 0,5 m/pixel)
- Cada patch bruto é reamostrado para HR=45×45 a 5 m/pixel
- Gera LR=15×15 a 30 m/pixel (downscale 6× do HR)

Filtro estrito: descarta patch se houver qualquer NaN ou valores >1e20 ou < -1e20
no LR ou no HR. Sem outros critérios de exclusão.
"""

import numpy as np
import tensorflow as tf
from osgeo import gdal
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)


def _read_region_geosampa(geosampa_path, x_start=18437, y_start=13802, region_width=6000, region_height=6000):
    ds = gdal.Open(geosampa_path)
    if ds is None:
        raise RuntimeError("Não foi possível abrir GEOSAMPA")

    band = ds.GetRasterBand(1)
    geosampa = band.ReadAsArray(x_start, y_start, region_width, region_height)
    if geosampa is None:
        raise RuntimeError("Não foi possível ler região do GEOSAMPA")

    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    ds = None

    if nodata is not None:
        valid_mask = (geosampa != nodata) & ~np.isnan(geosampa) & ~np.isinf(geosampa)
    else:
        valid_mask = ~np.isnan(geosampa) & ~np.isinf(geosampa)

    return geosampa.astype(np.float32), valid_mask.astype(np.uint8), gt


def _compute_raw_patch_params(geotransform, hr_size=45, lr_patch_size=15, target_px=5.0):
    """Converte tamanhos-alvo (em pixels a 5 m) para pixels na resolução nativa.

    Retorna: raw_hr_w, raw_hr_h, raw_stride_w, raw_stride_h
    """
    px_w = abs(float(geotransform[1]))
    px_h = abs(float(geotransform[5]))
    if px_w <= 0 or px_h <= 0:
        raise RuntimeError("Pixel size inválido no GEOSAMPA")

    # 45 px @5 m → largura/altura em pixels nativos
    raw_hr_w = max(1, int(round(hr_size * (target_px / px_w))))
    raw_hr_h = max(1, int(round(hr_size * (target_px / px_h))))
    # stride equivalente a 15 px @5 m
    raw_stride_w = max(1, int(round(lr_patch_size * (target_px / px_w))))
    raw_stride_h = max(1, int(round(lr_patch_size * (target_px / px_h))))
    return raw_hr_w, raw_hr_h, raw_stride_w, raw_stride_h


def create_real_lr_hr_pairs(anadem_path, geosampa_path, output_dir, patch_size=15):
    """
    Cria pares LR-HR REAIS usando GEOSAMPA com corte em resolução nativa:
    - Crop bruto → reamostra para HR=45×45 @5 m
    - LR=15×15 @30 m via downscale 6× do HR
    """

    print("🎯 CRIANDO DADOS (RAW CROP → HR 5 m → LR 30 m)")
    print("=" * 58)

    # 1) Ler região bruta do GEOSAMPA (0,5 m/pixel típico)
    x_start, y_start = 18437, 13802
    region_width, region_height = 6000, 6000
    print(f"📍 Região GEOSAMPA bruta: ({x_start}, {y_start}) - {region_width}x{region_height}")
    geosampa_raw, valid_raw, gt = _read_region_geosampa(geosampa_path, x_start, y_start, region_width, region_height)

    # 2) Tamanhos de crop na grade nativa para obter HR=45×45 @5 m
    hr_size = 45
    scale_factor = 3  # HR 45×45 → LR 15×15 (3× downscale)
    raw_hr_w, raw_hr_h, raw_stride_w, raw_stride_h = _compute_raw_patch_params(gt, hr_size=hr_size, lr_patch_size=patch_size, target_px=5.0)
    print(f"🧮 Crop bruto: {raw_hr_w}×{raw_hr_h} (stride {raw_stride_w}×{raw_stride_h}) → HR 45×45")

    # 3) Extrair crops na resolução nativa e reamostrar cada um para HR (5 m)
    patches_lr = []
    patches_hr = []
    H, W = geosampa_raw.shape
    tested = 0
    for y in range(0, H - raw_hr_h + 1, raw_stride_h):
        for x in range(0, W - raw_hr_w + 1, raw_stride_w):
            tested += 1
            raw_crop = geosampa_raw[y:y+raw_hr_h, x:x+raw_hr_w]

            # Reamostrar o crop bruto para HR=45×45 @5 m
            hr_patch = cv2.resize(raw_crop, (hr_size, hr_size), interpolation=cv2.INTER_AREA).astype(np.float32)

            # Gerar LR=15×15 a partir do HR
            lr_patch = create_realistic_lr_from_hr(hr_patch, scale_factor)

            # Filtro estrito: NaN/Inf já foram evitados pelo load; aqui aplica extremos
            if validate_real_patch_pair(lr_patch, hr_patch):
                patches_lr.append(lr_patch)
                patches_hr.append(hr_patch)

    print(f"🔍 Posições testadas: {tested}")
    print(f"✅ Pares aceitos: {len(patches_lr)}")

    if len(patches_lr) == 0:
        print("⚠️ Nenhum par válido. Nada a salvar.")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    for i, (lr, hr) in enumerate(zip(patches_lr, patches_hr)):
        np.save(os.path.join(output_dir, f"real_lr_{i:06d}.npy"), lr)
        np.save(os.path.join(output_dir, f"real_hr_{i:06d}.npy"), hr)

    return len(patches_lr)


def create_realistic_lr_from_hr(hr_patch, scale_factor):
    """Downsampling com blur + AREA para gerar LR (30 m)."""
    kernel_size = min(scale_factor, 15)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(hr_patch, (kernel_size, kernel_size), 0)
    lr_small = cv2.resize(blurred, (hr_patch.shape[1] // scale_factor, hr_patch.shape[0] // scale_factor), interpolation=cv2.INTER_AREA)
    noise_std = np.std(lr_small) * 0.02
    lr_patch = lr_small + np.random.normal(0, noise_std, lr_small.shape)
    return lr_patch.astype(np.float32)


def validate_real_patch_pair(lr_patch, hr_patch):
    # Verificação de shape e filtro estrito (NaN/extremos)
    if lr_patch.shape != (15, 15) or hr_patch.shape != (45, 45):
        return False
    invalid_hr = (
        np.isnan(hr_patch).any() or np.isinf(hr_patch).any() or
        (hr_patch > 1e20).any() or (hr_patch < -1e20).any()
    )
    invalid_lr = (
        np.isnan(lr_patch).any() or np.isinf(lr_patch).any() or
        (lr_patch > 1e20).any() or (lr_patch < -1e20).any()
    )
    return not (invalid_hr or invalid_lr)


def create_advanced_srcnn_model(input_shape=(15, 15, 1), scale_factor=3):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x1 = tf.keras.layers.Conv2D(64, (9, 9), activation='relu', padding='same')(inputs)
    x2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
    x3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Concatenate()([x1, x2, x3])
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D(size=(scale_factor, scale_factor), interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    output = tf.keras.layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=output)


def load_real_patches(patches_dir):
    lr_files = sorted([f for f in os.listdir(patches_dir) if f.startswith('real_lr_')])
    hr_files = sorted([f for f in os.listdir(patches_dir) if f.startswith('real_hr_')])
    lr_patches, hr_patches = [], []
    for lr_file, hr_file in zip(lr_files, hr_files):
        lr = np.load(os.path.join(patches_dir, lr_file))
        hr = np.load(os.path.join(patches_dir, hr_file))
        if lr.shape == (15, 15) and hr.shape == (45, 45):
            lr_patches.append(lr)
            hr_patches.append(hr)
    if len(lr_patches) == 0:
        raise ValueError("Nenhum patch válido encontrado (shape incorreto)")
    return np.array(lr_patches), np.array(hr_patches)


def robust_normalization(lr_patches, hr_patches):
    all_lr = np.concatenate([p.flatten() for p in lr_patches])
    all_hr = np.concatenate([p.flatten() for p in hr_patches])
    all_lr = all_lr[~np.isnan(all_lr) & ~np.isinf(all_lr)]
    all_hr = all_hr[~np.isnan(all_hr) & ~np.isinf(all_hr)]
    lr_min, lr_max = np.percentile(all_lr, [1, 99])
    hr_min, hr_max = np.percentile(all_hr, [1, 99])
    if lr_max - lr_min < 1e-6:
        lr_min, lr_max = np.min(all_lr), np.max(all_lr)
    if hr_max - hr_min < 1e-6:
        hr_min, hr_max = np.min(all_hr), np.max(all_hr)
    lr_norm = np.clip((lr_patches - lr_min) / (lr_max - lr_min), 0, 1)
    hr_norm = np.clip((hr_patches - hr_min) / (hr_max - hr_min), 0, 1)
    norm_params = {'lr_min': lr_min, 'lr_max': lr_max, 'hr_min': hr_min, 'hr_max': hr_max}
    print(f"📊 Normalização: LR [{lr_min:.2f}, {lr_max:.2f}], HR [{hr_min:.2f}, {hr_max:.2f}]")
    return lr_norm, hr_norm, norm_params


def stratified_split(lr_patches, hr_patches, test_size=0.2):
    variability = []
    for lr, hr in zip(lr_patches, hr_patches):
        variability.append(np.var(lr) + np.var(hr))
    variability = np.array(variability)
    quartiles = np.percentile(variability, [25, 50, 75])
    strata = np.zeros_like(variability)
    strata[variability <= quartiles[0]] = 0
    strata[(variability > quartiles[0]) & (variability <= quartiles[1])] = 1
    strata[(variability > quartiles[1]) & (variability <= quartiles[2])] = 2
    strata[variability > quartiles[2]] = 3
    return train_test_split(lr_patches, hr_patches, test_size=test_size, stratify=strata, random_state=42)


def create_and_train(patches_dir, epochs=15, batch_size=16):
    print("🧠 TREINAMENTO (RAW CROP) HR=5 m, LR=30 m")
    lr_p, hr_p = load_real_patches(patches_dir)
    lr_n, hr_n, norm_params = robust_normalization(lr_p, hr_p)
    lr_tr, lr_va, hr_tr, hr_va = stratified_split(lr_n, hr_n)

    # Data augmentation simples (flips/rotação 90°)
    def augment_pair(lr, hr):
        aug_lr, aug_hr = [lr], [hr]
        aug_lr.append(np.flip(lr, axis=0)); aug_hr.append(np.flip(hr, axis=0))
        aug_lr.append(np.flip(lr, axis=1)); aug_hr.append(np.flip(hr, axis=1))
        aug_lr.append(np.rot90(lr, k=1)); aug_hr.append(np.rot90(hr, k=1))
        return aug_lr, aug_hr

    lr_tr_aug, hr_tr_aug = [], []
    for lr, hr in zip(lr_tr, hr_tr):
        a_lr, a_hr = augment_pair(lr, hr)
        lr_tr_aug.extend(a_lr)
        hr_tr_aug.extend(a_hr)
    lr_tr = np.array(lr_tr_aug, dtype=np.float32)
    hr_tr = np.array(hr_tr_aug, dtype=np.float32)
    print(f"🔧 Augmentação: treino {lr_tr.shape[0]} patches")

    model = create_advanced_srcnn_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
                  loss='mse', metrics=['mae', 'mse'])
    os.makedirs('geospatial_output', exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('geospatial_output/advanced_model_5m_best.keras', monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger('geospatial_output/training_log_5m.csv', append=False)
    ]
    if epochs < 30:
        epochs = 30
    history = model.fit(lr_tr, hr_tr, validation_data=(lr_va, hr_va), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
    np.save('geospatial_output/norm_params_5m.npy', norm_params)
    model.save('geospatial_output/advanced_model_5m_last.keras')
    return model, history, norm_params


def main():
    print("🚀 TREINAMENTO (RAW CROP STRICT) 30 m → 5 m (3×)")
    print("=" * 70)
    print("1️⃣ Gerando pares LR-HR com CROP nativo → HR 5 m → LR 30 m...")
    num_patches = create_real_lr_hr_pairs(
        anadem_path='data/images/ANADEM_AricanduvaBufferUTM.tif',
        geosampa_path='data/images/MDTGeosampa_AricanduvaBufferUTM.tif',
        output_dir='geospatial_output/real_patches_30m_5m'
    )
    print("2️⃣ Treinando modelo (épocas=100)...")
    model, history, norm_params = create_and_train('geospatial_output/real_patches_30m_5m', epochs=100, batch_size=16)
    print("✅ Concluído!")
    print(f"📊 Patches: {num_patches}")
    print("📁 Modelo: geospatial_output/advanced_model_5m_best.keras")


if __name__ == "__main__":
    main()






