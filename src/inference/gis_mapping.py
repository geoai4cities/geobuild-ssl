import ssl  # noqa: precede sys.path
import os, sys, csv, json, math, random, re, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ssl"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "detection"))


# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════

CURRICULUM_ROOT = PROJECT_ROOT / "data" / "curriculum"
GT_LABELS_CSV   = PROJECT_ROOT / "data" / "annotations" / "clean_test_labels.csv"
PANO_GPS_CSV    = PROJECT_ROOT / "data" / "annotations" / "panorama_gps.csv"
OUTPUT_DIR      = PROJECT_ROOT / "results" / "gis"
MAPS_DIR        = OUTPUT_DIR / "maps"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)

# Real city centres
CITY_CENTERS = {
    "bglr":    (12.9716, 77.5946),   # Bangalore
    "chennai": (13.0827, 80.2707),
    "delhi":   (28.7041, 77.1025),
    "mumbai":  (19.0760, 72.8777),
    "osaka":   (34.6937, 135.5023),
    "tokyo":   (35.6762, 139.6503),
}

# Random radius around city centre for mock GPS (degrees)
GPS_OFFSET_RADIUS = 0.015  # ≈ 1.5km

CLASS_COLORS_HEX = {
    "residential":   "#4caf50",   # green
    "commercial":    "#2196f3",   # blue
    "industrial":    "#ff9800",   # orange
    "institutional": "#9c27b0",   # purple
}


# ════════════════════════════════════════════════════════════════════
# STEP 1: GENERATE MOCK GPS
# ════════════════════════════════════════════════════════════════════

def parse_filename(name):
    """Extract pano_id and yaw_deg from filename like '9_yaw_337p50_pitch_000p00.jpg'."""
    m = re.match(r"(\w+)_yaw_(\d{3})p(\d{2})_pitch_", name)
    if not m:
        return None, None
    pano_id = m.group(1)
    yaw = int(m.group(2)) + int(m.group(3)) / 100
    return pano_id, yaw


def generate_panorama_gps():
    """Assign mock GPS to every unique panorama, grouped by city."""
    random.seed(42)
    panoramas = {}  # (city, pano_id) → (lat, lon)

    for city in CITY_CENTERS:
        files = list(CURRICULUM_ROOT.rglob(f"*/{city}/*.jpg"))
        if not files:
            continue
        unique_panos = sorted({parse_filename(f.name)[0] for f in files
                               if parse_filename(f.name)[0] is not None})
        cy, cx = CITY_CENTERS[city]
        for pano_id in unique_panos:
            # Random offset within ~1.5km
            dlat = (random.random() - 0.5) * 2 * GPS_OFFSET_RADIUS
            dlon = (random.random() - 0.5) * 2 * GPS_OFFSET_RADIUS
            panoramas[(city, pano_id)] = (round(cy + dlat, 6),
                                          round(cx + dlon, 6))

    # Save CSV
    with open(PANO_GPS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city", "pano_id", "lat", "lon"])
        for (city, pano_id), (lat, lon) in panoramas.items():
            w.writerow([city, pano_id, lat, lon])
    print(f"[GPS] {len(panoramas)} panoramas → {PANO_GPS_CSV}")
    return panoramas


# ════════════════════════════════════════════════════════════════════
# STEP 2: GEOGRAPHIC PROJECTION
# ════════════════════════════════════════════════════════════════════

def project_building(cam_lat, cam_lon, bearing_deg, distance_m):
    """Project a building location given camera GPS + compass bearing + distance."""
    R = 6371000  # Earth radius in metres
    lat1 = math.radians(cam_lat)
    lon1 = math.radians(cam_lon)
    brng = math.radians(bearing_deg)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / R)
        + math.cos(lat1) * math.sin(distance_m / R) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(distance_m / R) * math.cos(lat1),
        math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


# ════════════════════════════════════════════════════════════════════
# STEP 3: RUN INFERENCE + PROJECT BUILDINGS
# ════════════════════════════════════════════════════════════════════

def run_inference_with_gps(panoramas):
    """Run pipeline on curriculum images, project each building to lat/lon."""
    from inference.geobuild_inference import GeoBuildPipeline, DEPTH_THRESHOLD

    pipeline = GeoBuildPipeline()
    print()

    images = sorted(CURRICULUM_ROOT.rglob("*.jpg"))
    print(f"[INFERENCE] Processing {len(images)} curriculum images")

    rows = []
    n_done = 0
    t0 = time.time()

    for img_path in images:
        pano_id, yaw = parse_filename(img_path.name)
        if pano_id is None:
            continue

        # Find city from path
        rel_parts = img_path.relative_to(CURRICULUM_ROOT).parts
        # Path: split / diff / country / city / file.jpg
        if len(rel_parts) >= 5:
            city = rel_parts[3]
        else:
            continue

        gps = panoramas.get((city, pano_id))
        if gps is None:
            continue
        cam_lat, cam_lon = gps

        # Inference
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            depth_map = pipeline.depth.estimate(img_pil)
            buildings = pipeline.detector.detect(img_pil, depth_map)

            for b in buildings:
                x1, y1, x2, y2 = b["bbox"]
                crop_bgr = img_bgr[y1:y2, x1:x2]
                if crop_bgr.size == 0:
                    continue
                crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                text_emb, ocr_texts = pipeline.ocr.extract(crop_bgr)
                depth_scalar = min(b["depth_m"] / DEPTH_THRESHOLD, 1.0)
                cls_result = pipeline.classifier.classify(crop_pil, depth_scalar, text_emb)

                # Project to lat/lon
                bld_lat, bld_lon = project_building(cam_lat, cam_lon, yaw, b["depth_m"])

                rows.append({
                    "image": str(img_path.relative_to(CURRICULUM_ROOT)),
                    "city": city,
                    "pano_id": pano_id,
                    "yaw_deg": round(yaw, 2),
                    "cam_lat": cam_lat,
                    "cam_lon": cam_lon,
                    "building_id": b["id"],
                    "depth_m": b["depth_m"],
                    "lat": round(bld_lat, 6),
                    "lon": round(bld_lon, 6),
                    "class_pred": cls_result["class"],
                    "confidence": cls_result["confidence"],
                    "ocr_texts": "|".join(ocr_texts[:3]),
                })
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            continue

        n_done += 1
        if n_done % 100 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (len(images) - n_done) / rate
            print(f"  [{n_done}/{len(images)}] {len(rows)} buildings, "
                  f"{rate:.1f} img/s, ETA {eta/60:.1f}min")

    # Save
    csv_path = OUTPUT_DIR / "buildings_geo.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[INFERENCE] {len(rows)} buildings → {csv_path}")
    return rows


# ════════════════════════════════════════════════════════════════════
# STEP 4: COMPARE WITH GROUND TRUTH
# ════════════════════════════════════════════════════════════════════

def compare_with_ground_truth(predictions):
    """Compare predicted classifications with ground truth labels.

    Ground truth is per-instance bbox in source image; we match by
    image + IoU between predicted bbox and GT bbox.
    """
    print("\n[COMPARE] Loading ground truth labels...")
    if not GT_LABELS_CSV.exists():
        print(f"  [WARN] No ground truth at {GT_LABELS_CSV}")
        return None

    # Load GT
    gt_by_image = defaultdict(list)
    with open(GT_LABELS_CSV) as f:
        for row in csv.DictReader(f):
            img = row.get("source_image", "")
            try:
                gt_by_image[img].append({
                    "x1": float(row["x1"]), "y1": float(row["y1"]),
                    "x2": float(row["x2"]), "y2": float(row["y2"]),
                    "class": row.get("final_usage", "unknown"),
                })
            except Exception:
                continue
    print(f"  [GT] {sum(len(v) for v in gt_by_image.values())} GT instances "
          f"in {len(gt_by_image)} images")

    # Match predictions to GT (currently we don't store pred bbox in GIS rows)
    # Aggregate per image instead — class distribution
    pred_by_image = defaultdict(list)
    for r in predictions:
        # Match via "data/curriculum/" + image rel path
        img_key = "data/curriculum/" + r["image"]
        pred_by_image[img_key].append(r["class_pred"])

    # Per-image majority vote comparison
    n_match = 0
    n_total = 0
    confusion = Counter()

    for img, gt_list in gt_by_image.items():
        if img not in pred_by_image:
            continue
        gt_classes = [g["class"] for g in gt_list]
        pred_classes = pred_by_image[img]
        if not gt_classes or not pred_classes:
            continue

        gt_majority = Counter(gt_classes).most_common(1)[0][0]
        pred_majority = Counter(pred_classes).most_common(1)[0][0]
        confusion[(gt_majority, pred_majority)] += 1
        if gt_majority == pred_majority:
            n_match += 1
        n_total += 1

    accuracy = n_match / max(n_total, 1)
    metrics = {
        "n_images_compared": n_total,
        "n_correct_majority": n_match,
        "image_majority_accuracy": round(accuracy, 4),
        "confusion_matrix": {f"{gt}→{pred}": count
                            for (gt, pred), count in confusion.items()},
    }
    with open(OUTPUT_DIR / "comparison_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [METRICS] Image-level majority accuracy: {accuracy:.2%} "
          f"({n_match}/{n_total})")
    return metrics


# ════════════════════════════════════════════════════════════════════
# STEP 5: VISUALIZE WITH FOLIUM
# ════════════════════════════════════════════════════════════════════

def make_folium_maps(predictions):
    """Create per-city interactive Folium maps + a combined map."""
    import folium
    from folium.plugins import MarkerCluster, HeatMap

    print("\n[MAPS] Generating Folium maps...")

    # Group by city
    by_city = defaultdict(list)
    for r in predictions:
        by_city[r["city"]].append(r)

    for city, rows in by_city.items():
        if not rows:
            continue
        center_lat = np.mean([r["lat"] for r in rows])
        center_lon = np.mean([r["lon"] for r in rows])
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14,
                       tiles="OpenStreetMap")

        # Marker cluster
        cluster = MarkerCluster(name="Buildings").add_to(m)
        for r in rows:
            color = CLASS_COLORS_HEX.get(r["class_pred"], "#888")
            popup = (f"<b>{r['class_pred'].capitalize()}</b><br>"
                     f"Confidence: {r['confidence']*100:.0f}%<br>"
                     f"Depth: {r['depth_m']}m<br>"
                     f"Pano: {r['pano_id']} (yaw {r['yaw_deg']}°)<br>"
                     f"OCR: {r['ocr_texts'] or 'none'}")
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.75,
                popup=folium.Popup(popup, max_width=300),
            ).add_to(cluster)

        # Heatmap layer per class
        for cls in CLASS_COLORS_HEX:
            cls_rows = [r for r in rows if r["class_pred"] == cls]
            if not cls_rows:
                continue
            HeatMap(
                [[r["lat"], r["lon"]] for r in cls_rows],
                name=f"{cls.capitalize()} heatmap",
                radius=15, blur=20, show=False,
            ).add_to(m)

        # Legend
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                    background: white; padding: 12px; border: 2px solid #333;
                    border-radius: 6px; font-family: sans-serif; font-size: 14px;">
        <b>Building Usage</b><br>
        """
        for cls, color in CLASS_COLORS_HEX.items():
            legend_html += (f'<i style="background:{color}; width:14px; '
                            f'height:14px; display:inline-block; '
                            f'margin-right:5px;"></i>{cls.capitalize()}<br>')
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl().add_to(m)
        out = MAPS_DIR / f"{city}.html"
        m.save(str(out))
        print(f"  [MAP] {city}: {len(rows)} buildings → {out}")

    # Combined world map
    if by_city:
        m = folium.Map(location=[20, 80], zoom_start=2, tiles="OpenStreetMap")
        for city, rows in by_city.items():
            cluster = MarkerCluster(name=city).add_to(m)
            for r in rows:
                color = CLASS_COLORS_HEX.get(r["class_pred"], "#888")
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=4, color=color, fill=True, fill_opacity=0.75,
                    popup=f"{r['class_pred']} {r['confidence']*100:.0f}% "
                          f"({city})",
                ).add_to(cluster)
        folium.LayerControl().add_to(m)
        m.save(str(MAPS_DIR / "all_cities.html"))
        print(f"  [MAP] all_cities → {MAPS_DIR / 'all_cities.html'}")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("GeoBuild-SSL → GIS Mapping Pipeline")
    print("=" * 60)

    panoramas = generate_panorama_gps()
    predictions = run_inference_with_gps(panoramas)
    metrics = compare_with_ground_truth(predictions)
    make_folium_maps(predictions)

    # Class distribution summary
    classes = Counter(r["class_pred"] for r in predictions)
    print(f"\n[SUMMARY] Class distribution:")
    for c, n in classes.most_common():
        print(f"  {c:15s}: {n} ({n / max(len(predictions), 1) * 100:.0f}%)")

    if metrics:
        print(f"\n[GROUND TRUTH] Image-majority accuracy: "
              f"{metrics['image_majority_accuracy']:.2%}")

    print(f"\n[DONE] All outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
