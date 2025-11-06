#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np

try:
	from insightface.app import FaceAnalysis
except Exception as e:
	raise RuntimeError(f"insightface is required: {e}")


def ensure_dir(p: Path):
	p.parent.mkdir(parents=True, exist_ok=True)


def enhance_image(img: np.ndarray) -> np.ndarray:
	# CLAHE on Y channel in YCrCb
	ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	y, cr, cb = cv2.split(ycc)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	y2 = clahe.apply(y)
	ycc2 = cv2.merge([y2, cr, cb])
	res = cv2.cvtColor(ycc2, cv2.COLOR_YCrCb2BGR)
	# mild denoise and sharpen
	den = cv2.bilateralFilter(res, d=7, sigmaColor=50, sigmaSpace=50)
	blur = cv2.GaussianBlur(den, (0,0), 1.0)
	sharp = cv2.addWeighted(den, 1.5, blur, -0.5, 0)
	return sharp


def detect_face(app: FaceAnalysis, img: np.ndarray):
	faces = app.get(img)
	if not faces:
		return None
	faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
	return faces[0]


def crop_and_segment(img: np.ndarray, bbox) -> tuple[np.ndarray, np.ndarray]:
	x1, y1, x2, y2 = map(int, bbox)
	x1 = max(0, x1); y1 = max(0, y1); x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
	crop = img[y1:y2, x1:x2].copy()
	# soft ellipse mask inside bbox
	mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
	center = (mask.shape[1]//2, mask.shape[0]//2)
	axes = (int(mask.shape[1]*0.45), int(mask.shape[0]*0.58))
	cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
	mask = cv2.GaussianBlur(mask, (31,31), 0)
	seg = crop.copy()
	for c in range(3):
		seg[:,:,c] = (seg[:,:,c] * (mask/255.0)).astype(np.uint8)
	return crop, seg


def main():
	parser = argparse.ArgumentParser(description='Preprocess: enhance, normalize, detect face, segment')
	parser.add_argument('image', help='Input image path')
	parser.add_argument('--out-dir', required=True, help='Output directory')
	parser.add_argument('--providers', nargs='+', default=['CPUExecutionProvider'])
	args = parser.parse_args()

	img_path = Path(args.image)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	img = cv2.imread(str(img_path))
	if img is None:
		raise FileNotFoundError(str(img_path))
	enh = enhance_image(img)
	cv2.imwrite(str(out_dir / 'enhanced.jpg'), enh)

	app = FaceAnalysis(name='buffalo_l', providers=args.providers)
	app.prepare(ctx_id=0 if 'CUDAExecutionProvider' in args.providers else -1, det_size=(640,640))
	face = detect_face(app, enh)
	if face is None:
		print('No face detected')
		return
	crop, seg = crop_and_segment(enh, face.bbox)
	cv2.imwrite(str(out_dir / 'crop.jpg'), crop)
	cv2.imwrite(str(out_dir / 'segment.jpg'), seg)
	print(str(out_dir))


if __name__ == '__main__':
	main()
