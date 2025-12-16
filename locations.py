LOCATION_INDEX = {}

def slugify(name):
    return name.lower().replace(" ", "-")

def register(group, parent, data):
    for name,(lat,lon) in data.items():
        LOCATION_INDEX[slugify(name)] = {
            "name": name,
            "group": group,
            "parent": parent,
            "lat": lat,
            "lon": lon
        }

# === DATA KAMU (PERSIS) ===
from raw_locations import DEFAULT_LOCATIONS, SULSEL, KECAMATAN

register("default","Nasional",DEFAULT_LOCATIONS)
register("sulsel","Sulawesi Selatan",SULSEL)

for kab, data in KECAMATAN.items():
    register("kecamatan", kab, data)
