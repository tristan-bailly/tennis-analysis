"""
run.py
------
Script simple et rapide pour lancer l'analyse tennis sur une ou toutes les vidéos.

Usage :
  python3 run.py 5                  # Analyse dataset/videos/5.mp4 et l'ouvre
  python3 run.py --video ma_video.mp4 # Analyse n'importe quelle vidéo
  python3 run.py --all              # Analyse TOUTES les vidéos du dossier dataset/videos/
  python3 run.py --list             # Liste toutes les vidéos disponibles
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add court directory to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "court"))
from yolo_and_tracknet_inference import run_inference


DEFAULT_TRACKNET = "weights/tracknet.pt"
DEFAULT_YOLO = "weights/best.pt"
VIDEOS_DIR = "dataset/videos"
OUTPUT_DIR = "output-videos"


def list_videos():
    print("\n📹 Vidéos disponibles dans dataset/videos/ :")
    print("-" * 55)
    videos = sorted([f for f in os.listdir(VIDEOS_DIR) if f.endswith((".mp4", ".mov", ".avi"))])
    for v in videos:
        p = os.path.join(VIDEOS_DIR, v)
        sz = os.path.getsize(p) / (1024 * 1024)
        print(f"  • {v:<15} ({sz:.1f} MB)  ->  python3 run.py {os.path.splitext(v)[0]}")
    print("-" * 55)


def process_video(video_path: str, output_path: str = None, open_after: bool = True):
    if not os.path.exists(video_path):
        print(f"❌ Erreur: vidéo introuvable : {video_path}")
        return False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_name = Path(video_path).stem
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, f"output{video_name}.mp4")

    print("\n" + "=" * 65)
    print(f"🎾 Lancement de l'analyse : {video_path}")
    print(f"📁 Fichier de sortie     : {output_path}")
    print("=" * 65)

    try:
        run_inference(
            video_path=video_path,
            tracknet_model_path=DEFAULT_TRACKNET,
            yolo_model_path=DEFAULT_YOLO,
            output_path=output_path,
            enable_interpolation=True,
            ball_smoothing=0.5,
            trace_length=9,
        )
        print(f"\n✅ Analyse terminée avec succès : {output_path}")

        if open_after:
            print("🎬 Ouverture de la vidéo dans QuickTime Player...")
            subprocess.run(["open", "-a", "QuickTime Player", output_path])
        return True
    except Exception as e:
        print(f"❌ Erreur lors du traitement de {video_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Tennis CV Analysis Runner")
    parser.add_argument("target", nargs="?", default=None,
                        help="Numéro ou nom de vidéo (ex: 5 ou dataset/videos/5.mp4)")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Chemin de la vidéo à analyser")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Analyser TOUTES les vidéos dans dataset/videos/")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Chemin du fichier de sortie")
    parser.add_argument("--no-open", action="store_true",
                        help="Ne pas ouvrir la vidéo après génération")
    parser.add_argument("--list", "-l", action="store_true",
                        help="Lister toutes les vidéos disponibles")

    args = parser.parse_args()

    if args.list:
        list_videos()
        return

    # Mode ALL : traite toutes les vidéos du dossier
    if args.all:
        videos = sorted([f for f in os.listdir(VIDEOS_DIR) if f.endswith((".mp4", ".mov", ".avi"))])
        print(f"\n🚀 Traitement de {len(videos)} vidéos dans {VIDEOS_DIR}/ ...")
        for i, v in enumerate(videos, 1):
            v_path = os.path.join(VIDEOS_DIR, v)
            out_path = os.path.join(OUTPUT_DIR, f"output{Path(v).stem}.mp4")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                print(f"[{i}/{len(videos)}] {v} déjà traité ({out_path}), passage au suivant.")
                continue
            print(f"\n[{i}/{len(videos)}] Traitement de {v}...")
            should_open = (i == len(videos)) and not args.no_open
            process_video(v_path, out_path, open_after=should_open)
        print(f"\n🎉 Toutes les {len(videos)} vidéos ont été traitées ! Dossier : {OUTPUT_DIR}/")
        subprocess.run(["open", OUTPUT_DIR])
        return

    # Déterminer la vidéo cible
    target = args.video or args.target
    if not target:
        list_videos()
        print("\n💡 Astuce : tape 'python3 run.py 5' pour analyser la vidéo 5, ou 'python3 run.py --all' pour toutes.")
        return

    # Si c'est juste un numéro (ex: '5' ou '8')
    if not os.path.exists(target):
        candidate = os.path.join(VIDEOS_DIR, f"{target}.mp4")
        if os.path.exists(candidate):
            target = candidate
        else:
            candidate = os.path.join(VIDEOS_DIR, target)
            if os.path.exists(candidate):
                target = candidate

    process_video(target, args.output, open_after=not args.no_open)


if __name__ == "__main__":
    main()
