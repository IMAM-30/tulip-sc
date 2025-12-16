# raw_locations.py - DIPERBARUI SESUAI LARAVEL
# Data lokasi sesuai dengan array api1Locations di Laravel

DEFAULT_LOCATIONS = {
    # Jakarta
    "Jakarta Selatan": (-6.261, 106.810),
    "Jakarta Timur": (-6.225, 106.900),
    "Jakarta Utara": (-6.138, 106.862),
    "Jakarta Pusat": (-6.182, 106.840),
    "Jakarta Barat": (-6.167, 106.767),
    
    # Jawa Barat
    "Sukabumi": (-6.9277, 106.93),
    "Bekasi": (-6.238, 106.9756),
    "Bogor": (-6.5971, 106.8060),
    "Cianjur": (-6.818, 107.140),
}

SULSEL = {
    # Kota
    "Parepare": (-4.0096, 119.6236),
    "Makassar": (-5.1477, 119.4327),
    "Palopo": (-2.9925, 120.1969),
    
    # Kabupaten
    "Luwu Timur": (-2.6, 120.5),
    "Sinjai": (-5.1167, 120.25),
    "Pangkep": (-4.8333, 119.5833),
    "Luwu": (-3.3833, 120.3667),
    "Duampanua": (-3.92, 119.85),
    "Pinrang": (-3.7833, 119.65),
    "Soreang": (-4.005, 119.628),
    "Barru": (-4.4333, 119.6333),
    "Takalar": (-5.4167, 119.4833),
    "Balusu": (-3.45, 119.85),
    "Soppeng": (-4.3833, 119.9167),
    "Bacukiki Barat": (-4.02, 119.62),
    "Bacukiki": (-4.015, 119.625),
    "Watang Sawitto": (-3.78, 119.65),
    "Enrekang": (-3.55, 119.75),
    "Sidrap": (-3.9333, 119.9),
    "Tana Toraja": (-3.0833, 119.75),
    "Bone": (-4.5397, 120.3287),
    "Gowa": (-5.3167, 119.75),
    "Cempa": (-4.38, 119.92),
    "Mallusetasi": (-3.45, 119.85),  # Approx, adjust if needed
    "Toraja Utara": (-2.8333, 119.8333),
    "Batulappa": (-3.88, 119.78),
    "Lanrisang": (-3.8, 119.7),
    "Luwu Utara": (-2.7, 120.1),
    "Jeneponto": (-5.6333, 119.7333),
    "Bulukumba": (-5.5523, 120.1969),
    "Ujung": (-4.009, 119.623),
    "Maros": (-5.05, 119.5833),
    "Wajo": (-4.0333, 120.1833),
    "Tanete Rilau": (-4.47, 119.57),
}

# Data Kecamatan (untuk detail lebih lanjut)
KECAMATAN = {
    "Parepare": {
        "Soreang": (-4.005, 119.628),
        "Bacukiki": (-4.015, 119.625),
        "Bacukiki Barat": (-4.02, 119.62),
        "Ujung": (-4.009, 119.623),
    }
}

print(f"✅ Loaded locations: {len(DEFAULT_LOCATIONS)} default, {len(SULSEL)} Sulsel")