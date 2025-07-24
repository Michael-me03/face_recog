#!/usr/bin/env python3
"""
Product Recognition on Store Shelves
Erweiterte Version für 11 Models (0-10) und 11 Scenes (e1-e11)

Basiert auf dem Original Notebook von Baraghini Nicholas und Marini Luca
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys
import json
from datetime import datetime

def distance_2_points(A, B):
    """Berechnet Euclidische Distanz zwischen zwei Punkten"""
    return math.sqrt(np.power(A[0] - B[0], 2) + np.power(A[1] - B[1], 2))

def init_sift():
    """Initialisiert SIFT Detektor (Version-sicher)"""
    try:
        sift = cv2.SIFT_create()
        print("✓ SIFT (modern) initialisiert")
        return sift
    except AttributeError:
        try:
            sift = cv2.xfeatures2d.SIFT_create()
            print("✓ SIFT (xfeatures2d) initialisiert")
            return sift
        except AttributeError:
            raise Exception("SIFT nicht verfügbar. Installiere opencv-contrib-python")

def Matching(Model_Descriptors, Scene_Descriptors, Treshold=0.45, k=2):
    """
    Findet Matches zwischen Model und Scene Descriptors mit FLANN
    """
    if Model_Descriptors is None or Scene_Descriptors is None:
        return []
        
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(Model_Descriptors, Scene_Descriptors, k)
    except cv2.error:
        return []
    
    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < Treshold * n.distance:
                good.append(m)
                
    return good

def solve_exceeding_dimensions_of_bounding_boxes_in_scene(final_corners_of_bounding_boxes, scene_shape):
    """
    Korrigiert Bounding Boxes die über die Bildgrenzen hinausgehen
    """
    final_corners_of_bounding_boxes_without_exceeding_dimensions = []
    scene_height, scene_width = scene_shape[:2]

    for [fin_top_left_corner, fin_bottom_right_corner] in final_corners_of_bounding_boxes:
        fin_top_left_corner = list(fin_top_left_corner)
        fin_bottom_right_corner = list(fin_bottom_right_corner)
        
        # Grenzen korrigieren
        fin_top_left_corner[0] = max(0, min(fin_top_left_corner[0], scene_width))
        fin_top_left_corner[1] = max(0, min(fin_top_left_corner[1], scene_height))
        fin_bottom_right_corner[0] = max(0, min(fin_bottom_right_corner[0], scene_width))
        fin_bottom_right_corner[1] = max(0, min(fin_bottom_right_corner[1], scene_height))

        fin_top_left_corner = tuple(fin_top_left_corner)
        fin_bottom_right_corner = tuple(fin_bottom_right_corner)
        
        final_corners_of_bounding_boxes_without_exceeding_dimensions.append([
            (int(fin_top_left_corner[0]), int(fin_top_left_corner[1])), 
            (int(fin_bottom_right_corner[0]), int(fin_bottom_right_corner[1]))
        ])
    
    return final_corners_of_bounding_boxes_without_exceeding_dimensions

def split_image_into_N_x_M_bins_with_intensity_means(image, n_bins_width=3, n_bins_heigth=4):
    """
    Teilt ein Bild in N x M Bins auf und berechnet Farbmittelwerte
    """
    img_N_x_M_bins = {}
    
    if image is None or image.size == 0:
        return img_N_x_M_bins
    
    img_width = image.shape[1]
    img_height = image.shape[0]
    
    if img_height == 0 or img_width == 0:
        return img_N_x_M_bins
    
    step_width = int(img_width / n_bins_width)
    step_height = int(img_height / n_bins_heigth)
    
    if step_width == 0 or step_height == 0:
        return img_N_x_M_bins
    
    r = 0
    for row in np.arange(0, img_height, step_height):
        c = 0
        for col in np.arange(0, img_width, step_width):
            if row + 2 * step_height > img_height and col + 2 * step_width > img_width:
                partial_r_channel, partial_g_channel, partial_b_channel = cv2.split(image[row:, col:])
            elif row + 2 * step_height > img_height:
                partial_r_channel, partial_g_channel, partial_b_channel = cv2.split(image[row:, col : col + step_width])
            elif col + 2 * step_width > img_width:
                partial_r_channel, partial_g_channel, partial_b_channel = cv2.split(image[row : row + step_height, col :])
            else:
                partial_r_channel, partial_g_channel, partial_b_channel = cv2.split(
                    image[row : row + step_height, col : col + step_width])

            if r < n_bins_heigth and c < n_bins_width:
                img_N_x_M_bins[r, c] = (np.mean(partial_r_channel), np.mean(partial_g_channel), np.mean(partial_b_channel))

            c += 1
        r += 1
    
    return img_N_x_M_bins

def solve_color_problem_with_N_x_M_bins(final_corners_of_bounding_boxes_without_exceeding_dimensions, 
                                    model_img, scene_img, N=3, M=4, 
                                    COLOR_DIFF_IN_SINGLE_CHANNEL_TRES=79,
                                    MAX_NUM_OF_NO_GOOD_CELLS=3):
    """
    Farbbasierte Filterung mit N x M Bins
    """
    final_corners_of_bounding_boxes_after_color_problem = []
    
    for [fin_top_left_corner, fin_bottom_right_corner] in final_corners_of_bounding_boxes_without_exceeding_dimensions:
        
        means_bins_model_img = split_image_into_N_x_M_bins_with_intensity_means(model_img, N, M)
        
        scene_crop = scene_img[fin_top_left_corner[1]:fin_bottom_right_corner[1], 
                              fin_top_left_corner[0]:fin_bottom_right_corner[0]]
        
        if scene_crop.size == 0:
            continue
            
        means_bins_scene_img = split_image_into_N_x_M_bins_with_intensity_means(scene_crop, N, M)
    
        if not means_bins_scene_img or not means_bins_model_img:
            continue
    
        num_of_no_good = 0
        
        # Vergleiche die Bins (ohne oberste und unterste Reihe)
        for k, v in means_bins_model_img.items():
            if k[0] != 0 and k[0] != N:
                means_k_scene = means_bins_scene_img.get(k)
                if means_k_scene is None:
                    continue
                    
                diff_r = np.absolute(v[0] - means_k_scene[0])
                diff_g = np.absolute(v[1] - means_k_scene[1])
                diff_b = np.absolute(v[2] - means_k_scene[2])

                if (diff_r >= COLOR_DIFF_IN_SINGLE_CHANNEL_TRES or 
                    diff_g >= COLOR_DIFF_IN_SINGLE_CHANNEL_TRES or 
                    diff_b >= COLOR_DIFF_IN_SINGLE_CHANNEL_TRES):
                    num_of_no_good += 1
        
        if num_of_no_good <= MAX_NUM_OF_NO_GOOD_CELLS:
            final_corners_of_bounding_boxes_after_color_problem.append([
                (int(fin_top_left_corner[0]), int(fin_top_left_corner[1])), 
                (int(fin_bottom_right_corner[0]), int(fin_bottom_right_corner[1]))
            ])
            
    return final_corners_of_bounding_boxes_after_color_problem

def plot_final_bounding_boxes(img, final_corners_of_bounding_boxes, model_id, thickness=5):
    """
    Zeichnet finale Bounding Boxes mit Labels auf das Bild
    """
    result_img = img.copy()
    
    # Farben für verschiedene Models
    colors = [
        (0, 255, 0),    # Grün
        (255, 0, 0),    # Rot  
        (0, 0, 255),    # Blau
        (255, 255, 0),  # Gelb
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Lila
        (255, 165, 0),  # Orange
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
        (128, 0, 0),    # Maroon
    ]
    
    color = colors[model_id % len(colors)]
    
    for top_left_corner, bottom_right_corner in final_corners_of_bounding_boxes:
        cv2.rectangle(result_img, top_left_corner, bottom_right_corner, color, thickness)
        
        # Label hinzufügen
        label = f"Model {model_id}"
        cv2.putText(result_img, label, 
                   (top_left_corner[0], top_left_corner[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return result_img

def load_image_safely(paths):
    """
    Lädt ein Bild von verschiedenen möglichen Pfaden
    """
    for path in paths:
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                return img, path
    return None, None

def load_all_models(max_models=11):
    """
    Lädt alle verfügbaren Models (0 bis max_models-1)
    """
    models = {}
    sift = init_sift()
    
    print(f"\n=== LADE {max_models} MODELS ===")
    
    for i in range(max_models):
        model_paths = [
            f'./models/{i}.jpg',
            f'./models/{i}.png',
            f'models/{i}.jpg',
            f'models/{i}.png',
            f'{i}.jpg',
            f'{i}.png'
        ]
        
        model_img, model_path = load_image_safely(model_paths)
        
        if model_img is not None:
            # SIFT Features berechnen
            kp_model, des_model = sift.detectAndCompute(model_img, None)
            
            if des_model is not None and len(kp_model) > 0:
                models[i] = {
                    'image': model_img,
                    'keypoints': kp_model,
                    'descriptors': des_model,
                    'path': model_path,
                    'num_features': len(kp_model)
                }
                print(f"✓ Model {i}: {len(kp_model)} keypoints - {model_path}")
            else:
                print(f"✗ Model {i}: Keine Features gefunden - {model_path}")
        else:
            print(f"✗ Model {i}: Nicht gefunden")
    
    print(f"✓ {len(models)} Models erfolgreich geladen")
    return models

def load_all_scenes(max_scenes=11):
    """
    Lädt alle verfügbaren Scenes (e1 bis e{max_scenes})
    """
    scenes = {}
    sift = init_sift()
    
    print(f"\n=== LADE {max_scenes} SCENES ===")
    
    for i in range(1, max_scenes + 1):
        scene_paths = [
            f'./scenes/e{i}.jpg',
            f'./scenes/e{i}.png',
            f'scenes/e{i}.jpg',
            f'scenes/e{i}.png',
            f'e{i}.jpg',
            f'e{i}.png'
        ]
        
        scene_img, scene_path = load_image_safely(scene_paths)
        
        if scene_img is not None:
            # SIFT Features berechnen
            kp_scene, des_scene = sift.detectAndCompute(scene_img, None)
            
            if des_scene is not None and len(kp_scene) > 0:
                scenes[i] = {
                    'image': scene_img,
                    'keypoints': kp_scene,
                    'descriptors': des_scene,
                    'path': scene_path,
                    'num_features': len(kp_scene)
                }
                print(f"✓ Scene e{i}: {len(kp_scene)} keypoints - {scene_path}")
            else:
                print(f"✗ Scene e{i}: Keine Features gefunden - {scene_path}")
        else:
            print(f"✗ Scene e{i}: Nicht gefunden")
    
    print(f"✓ {len(scenes)} Scenes erfolgreich geladen")
    return scenes

def recognize_products_in_scene(models, scene_data, scene_id):
    """
    Erkennt alle Models in einer Scene
    """
    # Parameter
    MATCHING_THRESHOLD = 0.45
    MIN_NUM_OF_MATCHES = 15
    COLOR_DIFF_THRESHOLD = 79
    N_BINS_WIDTH = 3
    M_BINS_HEIGHT = 4
    
    scene_img = scene_data['image']
    kp_scene = scene_data['keypoints']
    des_scene = scene_data['descriptors']
    
    results = []
    result_img = scene_img.copy()
    
    print(f"\n=== ERKENNE PRODUKTE IN SCENE e{scene_id} ===")
    
    for model_id, model_data in models.items():
        print(f"\nTeste Model {model_id}...")
        
        # Feature Matching
        good_matches = Matching(model_data['descriptors'], des_scene, MATCHING_THRESHOLD)
        
        if len(good_matches) < MIN_NUM_OF_MATCHES:
            print(f"  ✗ Zu wenige Matches: {len(good_matches)}/{MIN_NUM_OF_MATCHES}")
            continue
        
        print(f"  ✓ {len(good_matches)} gute Matches gefunden")
        
        # Homographie berechnen
        src_pts = np.float32([model_data['keypoints'][m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            print(f"  ✗ Homographie fehlgeschlagen")
            continue
        
        # Bounding Box berechnen
        h, w = model_data['image'].shape[:2]
        corners = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        transformed_corners = cv2.perspectiveTransform(corners, M)
        
        top_left = transformed_corners[0][0]
        bottom_right = transformed_corners[2][0]
        
        # Dimensionen validieren
        width = int(distance_2_points(transformed_corners[0][0], transformed_corners[3][0]))
        height = int(distance_2_points(transformed_corners[0][0], transformed_corners[1][0]))
        
        if height <= width:
            print(f"  ✗ Ungültige Bounding Box Form: {width}x{height}")
            continue
        
        # Bounding Box korrigieren
        corners_list = [[top_left, bottom_right]]
        corrected_corners = solve_exceeding_dimensions_of_bounding_boxes_in_scene(
            corners_list, scene_img.shape
        )
        
        # Farbfilterung
        final_corners = solve_color_problem_with_N_x_M_bins(
            corrected_corners, model_data['image'], scene_img,
            N=N_BINS_WIDTH, M=M_BINS_HEIGHT,
            COLOR_DIFF_IN_SINGLE_CHANNEL_TRES=COLOR_DIFF_THRESHOLD
        )
        
        if len(final_corners) > 0:
            print(f"  ✓ ERKANNT! Position: {final_corners[0][0]}, Größe: {width}x{height}")
            
            # Ergebnis speichern
            results.append({
                'model_id': model_id,
                'position': final_corners[0][0],
                'size': (width, height),
                'confidence': len(good_matches),
                'corners': final_corners[0]
            })
            
            # Bounding Box zeichnen
            result_img = plot_final_bounding_boxes(result_img, final_corners, model_id)
        else:
            print(f"  ✗ Farbfilterung fehlgeschlagen")
    
    return results, result_img

def main():
    """
    Hauptfunktion - Multi-Product Recognition
    """
    print("=" * 80)
    print("MULTI-PRODUCT RECOGNITION: 11 Models × 11 Scenes")
    print("=" * 80)
    
    # Kommandozeilen-Parameter
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("Usage:")
            print("  python main.py                    # Teste alle Models in allen Scenes")
            print("  python main.py --model 0          # Teste nur Model 0 in allen Scenes")
            print("  python main.py --scene 1          # Teste alle Models nur in Scene e1")
            print("  python main.py --model 0 --scene 1 # Teste nur Model 0 in Scene e1")
            return
    
    # ================================================================
    # SCHRITT 1: Alle Models laden
    # ================================================================
    
    models = load_all_models(6)
    if not models:
        print("✗ Keine Models konnten geladen werden!")
        return
    
    # ================================================================
    # SCHRITT 2: Alle Scenes laden
    # ================================================================
    
    scenes = load_all_scenes(7)
    if not scenes:
        print("✗ Keine Scenes konnten geladen werden!")
        return
    
    # ================================================================
    # SCHRITT 3: Parameter aus Kommandozeile
    # ================================================================
    
    test_models = list(models.keys())
    test_scenes = list(scenes.keys())
    
    # Parse Kommandozeilen-Argumente
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--model' and i + 1 < len(sys.argv):
            model_id = int(sys.argv[i + 1])
            if model_id in models:
                test_models = [model_id]
                print(f"✓ Teste nur Model {model_id}")
            else:
                print(f"✗ Model {model_id} nicht verfügbar")
                return
            i += 2
        elif sys.argv[i] == '--scene' and i + 1 < len(sys.argv):
            scene_id = int(sys.argv[i + 1])
            if scene_id in scenes:
                test_scenes = [scene_id]
                print(f"✓ Teste nur Scene e{scene_id}")
            else:
                print(f"✗ Scene e{scene_id} nicht verfügbar")
                return
            i += 2
        else:
            i += 1
    
    # ================================================================
    # SCHRITT 4: Produkterkennung durchführen
    # ================================================================
    
    all_results = {}
    
    for scene_id in test_scenes:
        print(f"\n{'='*60}")
        print(f"VERARBEITE SCENE e{scene_id}")
        print(f"{'='*60}")
        
        # Filtere Models für diese Scene
        scene_models = {k: v for k, v in models.items() if k in test_models}
        
        # Erkenne Produkte
        results, result_img = recognize_products_in_scene(scene_models, scenes[scene_id], scene_id)
        
        all_results[scene_id] = {
            'scene_path': scenes[scene_id]['path'],
            'products_found': len(results),
            'results': results,
            'result_image': result_img
        }
        
        # Ergebnisse anzeigen
        if results:
            print(f"\n✓ {len(results)} Produkte in Scene e{scene_id} erkannt:")
            for result in results:
                print(f"  • Model {result['model_id']}: {result['position']} ({result['confidence']} matches)")
            
            # Visualisierung
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title(f'Scene e{scene_id}: {len(results)} Produkte erkannt')
            plt.axis('off')
            plt.show()
        else:
            print(f"\n✗ Keine Produkte in Scene e{scene_id} erkannt")
    
    # ================================================================
    # SCHRITT 5: Zusammenfassung und Export
    # ================================================================
    
    print(f"\n{'='*80}")
    print("GESAMTZUSAMMENFASSUNG")
    print(f"{'='*80}")
    
    total_detections = 0
    for scene_id, data in all_results.items():
        detections = data['products_found']
        total_detections += detections
        print(f"Scene e{scene_id}: {detections} Produkte")
    
    print(f"\nGesamt: {total_detections} Produkterkennungen")
    print(f"Getestete Models: {test_models}")
    print(f"Getestete Scenes: {test_scenes}")
    
    # JSON Export
    export_data = {}
    for scene_id, data in all_results.items():
        export_data[f"scene_e{scene_id}"] = {
            'scene_path': data['scene_path'],
            'products_found': data['products_found'],
            'detections': [
                {
                    'model_id': r['model_id'],
                    'position': r['position'],
                    'size': r['size'],
                    'confidence': r['confidence']
                }
                for r in data['results']
            ]
        }
    
    # Speichere Ergebnisse
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"recognition_results_{timestamp}.json"
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Ergebnisse gespeichert: {json_filename}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()